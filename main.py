import sys
import time
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tracker import AirHockeyTracker
from utils import resource_path

# When running as a PyInstaller bundle, use the non-interactive Agg backend
# so the plot is saved to a file without needing a display.
if getattr(sys, 'frozen', False):
    matplotlib.use('Agg')

DEBUG  = False  # set True to show contour-debug window
RECORD = False  # set True to record warped output to demo_output.mp4


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def put_text(img, text, pos, font_scale=0.7, color=(255, 255, 255), thickness=1):
    """Draws text with a black outline so it is readable on any background."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)


def print_summary(stats):
    """Prints a prediction error summary table to the console."""
    print("\n=== Trajectory Prediction Evaluation ===")
    print(f"{'Horizon':>10} | {'Mean error':>12} | {'Std dev':>8} | {'Samples':>8}")
    print("-" * 50)
    for h, stat in stats.items():
        if stat:
            print(f"{h:>8}f  | {stat['mean']:>10.1f} px | {stat['std']:>6.1f} px | {stat['count']:>8}")
        else:
            print(f"{h:>8}f  | {'no data':>12} | {'':>8} | {'':>8}")
    print()


def plot_results(ts_frames, ts_errors, accumulated_stats, fps_avg):
    """Két paneles hibaelemzési ábrát készít és ment el.

    Bal panel  – gördülő átlag hiba időben (vonaldiagram).
    Jobb panel – teljes videó átlag ± szórás horizononként (oszlopdiagram).
    """
    if not ts_frames:
        print("Nincs elegendő adat a grafikonhoz.")
        return

    colors = {5: '#2196F3', 10: '#FF9800', 20: '#F44336'}
    labels = {
        5:  f'5 képkocka  (~{5  / fps_avg * 1000:.0f} ms)',
        10: f'10 képkocka (~{10 / fps_avg * 1000:.0f} ms)',
        20: f'20 képkocka (~{20 / fps_avg * 1000:.0f} ms)',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Trajektória-predikció kiértékelése', fontsize=14, fontweight='bold')

    # --- Bal: gördülő átlag hiba időben ---
    for h, color in colors.items():
        vals = [e[h] for e in ts_errors if h in e]
        frms = [ts_frames[i] for i, e in enumerate(ts_errors) if h in e]
        if vals:
            ax1.plot(frms, vals, color=color, label=labels[h],
                     linewidth=1.5, alpha=0.85)

    ax1.set_xlabel('Képkocka sorszám')
    ax1.set_ylabel('Gördülő átlag hiba (px)')
    ax1.set_title('Predikciós hiba időben\n(gördülő átlag, ~60 képkocka ablak)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # --- Jobb: teljes videó összesítő ---
    valid_h = [h for h in (5, 10, 20) if accumulated_stats.get(h)]
    if valid_h:
        means  = [accumulated_stats[h]['mean']  for h in valid_h]
        stds   = [accumulated_stats[h]['std']   for h in valid_h]
        counts = [accumulated_stats[h]['count'] for h in valid_h]
        bcols  = [colors[h] for h in valid_h]

        bars = ax2.bar([labels[h] for h in valid_h], means,
                       yerr=stds, capsize=8, color=bcols, alpha=0.8,
                       error_kw={'linewidth': 2})

        pad = max(max(means) * 0.05, max(stds) * 0.15, 1.5)
        for bar, mean, std, count in zip(bars, means, stds, counts):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + pad,
                f'{mean:.1f} px\n(n={count})',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
            )

        ax2.set_ylabel('Átlag hiba (px)')
        ax2.set_title('Teljes videó összesítő\n(átlag ± szórás)')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(bottom=0)
    else:
        ax2.text(0.5, 0.5, 'Nincs elegendő adat', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=13)

    plt.tight_layout()
    out_path = 'prediction_error.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Grafikon mentve: {out_path}")

    # plt.show() is skipped when running as a compiled executable
    if not getattr(sys, 'frozen', False):
        try:
            plt.show()
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    video_path = resource_path('test_video.mp4')
    tracker    = AirHockeyTracker(video_path)
    print("Tracking started. Press 'q' to quit.")

    # Optional video recording
    writer  = None
    src_fps = tracker.cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    if RECORD:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('demo_output.mp4', fourcc, src_fps, (800, 400))
        print("Recording demo_output.mp4...")

    prev_time = time.time()
    fps_vals  = []
    ts_frames = []
    ts_errors = []

    while True:
        ret, frame = tracker.cap.read()
        if not ret:
            print("End of video stream.")
            break

        # --- Processing pipeline ---
        warped_frame, red_mask, combined_mask = tracker.preprocess_frame(frame)

        if DEBUG:
            tracker.debug_contours(warped_frame, red_mask)

        valid_candidates, paddles = tracker.detect_objects(red_mask, combined_mask)
        center                    = tracker.track_puck(valid_candidates)
        pred_pts                  = tracker.predict_trajectory(paddles)

        tracker.draw_elements(warped_frame, center, paddles, pred_pts)

        # --- FPS overlay (top-left) ---
        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now
        fps_vals.append(fps)
        put_text(warped_frame, f"FPS: {fps:.1f}", (10, 25),
                 font_scale=0.7, color=(255, 255, 0))

        # --- Tracking state overlay (top-left) ---
        if not tracker.kalman_initialized:
            state_text, state_color = "Initializing...", (0, 165, 255)
        elif tracker.lost_frames > 0:
            state_text, state_color = "Lost",            (0,   0, 255)
        else:
            state_text, state_color = "Tracking",        (0, 220,   0)
        put_text(warped_frame, state_text, (10, 55),
                 font_scale=0.7, color=state_color)

        # --- Record frame if enabled ---
        if writer:
            writer.write(warped_frame)

        # --- Prediction error data collection ---
        stats = tracker.get_error_stats(horizons=(5, 10, 20))
        row = {}
        for h, stat in stats.items():
            if stat and stat['count'] >= 5:
                row[h] = stat['mean']
        if row:
            ts_frames.append(tracker.frame_counter)
            ts_errors.append(row)

        tracker.frame_counter += 1

        cv2.imshow('Original', frame)
        cv2.imshow('Warped Table (Homography)', warped_frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    tracker.cap.release()
    if writer:
        writer.release()
        print("Demo video saved: demo_output.mp4")
    cv2.destroyAllWindows()

    # --- Post-run summary and plot ---
    acc_stats = tracker.get_accumulated_stats()
    fps_avg   = float(np.mean(fps_vals)) if fps_vals else src_fps
    print_summary(acc_stats)
    print(f"Average FPS: {fps_avg:.1f}")
    plot_results(ts_frames, ts_errors, acc_stats, fps_avg)


if __name__ == "__main__":
    main()
