import os
import cv2
import numpy as np
from collections import deque
from utils import resource_path


class AirHockeyTracker:

    #Tracking state machine
    INIT_CONFIRMATIONS = 3   # consecutive detections required before tracking starts
    MAX_LOST_FRAMES    = 8   # frames without detection before full reset
    MIN_STABLE_FRAMES  = 5   # stable frames required before trajectory is drawn

    #Physics constants
    FRICTION           = 0.995   # velocity multiplier per simulation step
    WALL_RESTITUTION   = 0.85    # energy retained after a wall bounce
    PADDLE_RESTITUTION = 0.80    # energy retained after a paddle bounce

    #Detection thresholds
    PUCK_AREA_MIN      = 150
    PUCK_AREA_MAX      = 1500
    PUCK_CIRCULARITY   = 0.3
    PADDLE_AREA_MIN    = 1000
    PADDLE_AREA_MAX    = 15000
    PADDLE_CIRCULARITY = 0.3
    PUCK_RADIUS        = 15      # pixels in warped (800x400) space

    #Table dimensions in warped space
    TABLE_W = 800
    TABLE_H = 400

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap        = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Error: cannot open video '{video_path}'.")

        # Tracking state
        self.init_buffer   = []   # accumulates detections during initialization phase
        self.lost_frames   = 0
        self.stable_frames = 0
        self.pts           = deque(maxlen=20)  # history of Kalman-filtered positions

        # Velocity smoother for trajectory prediction
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0

        # Evaluation bookkeeping
        self.frame_counter      = 0
        self.pred_history       = deque(maxlen=60)  # (frame_idx, [all sim. positions])
        self.pos_history        = deque(maxlen=60)  # (frame_idx, (cx, cy))
        self.accumulated_errors = {5: [], 10: [], 20: []}  # lifetime error log
        self._evaluated_pairs   = set()              # prevents double-counting

        # Background subtractor
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )

        self._init_homography()
        self._warmup_background(num_frames=120)
        self._init_kalman()

    # Initialisation helpers
    def _init_homography(self):
        """Loads the perspective transform matrix from file or falls back to defaults."""
        self.pts_dst = np.array([
            [0,            0],
            [self.TABLE_W, 0],
            [self.TABLE_W, self.TABLE_H],
            [0,            self.TABLE_H],
        ], dtype=np.float32)

        cal_path = resource_path("calibration.npy")
        if os.path.exists(cal_path):
            loaded = np.load(cal_path)
            if loaded.shape == (4, 2):
                self.pts_src = loaded.astype(np.float32)
                print("Calibration loaded from file.")
            else:
                print(f"Warning: calibration.npy has shape {loaded.shape}, expected (4,2). "
                      "Using default points.")
                self._use_default_pts()
        else:
            print("No calibration file found. Using hardcoded default points.")
            self._use_default_pts()

        self.matrix = cv2.getPerspectiveTransform(self.pts_src, self.pts_dst)

    def _use_default_pts(self):
        """Falls back to hardcoded source corners when no calibration file exists."""
        self.pts_src = np.array([
            [ 14,  27],
            [951,  21],
            [945, 513],
            [ 15, 519],
        ], dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.pts_src, self.pts_dst)

    def _warmup_background(self, num_frames=120):
        """Pre-trains the MOG2 background model before the main tracking loop."""
        print(f"Background model warmup ({num_frames} frames)...")
        warmup_cap = cv2.VideoCapture(self.video_path)

        for i in range(num_frames):
            ret, frame = warmup_cap.read()
            if not ret:
                print(f"  Video shorter than {num_frames} frames; warmup stopped at frame {i}.")
                break
            warped  = cv2.warpPerspective(frame, self.matrix, (self.TABLE_W, self.TABLE_H))
            blurred = cv2.GaussianBlur(warped, (11, 11), 0)
            self.backSub.apply(blurred, learningRate=0.05)

        warmup_cap.release()
        print("Background model ready.")

    def _init_kalman(self):
        """Creates and configures a 4-state (x, y, vx, vy) Kalman filter."""
        self.kalman = cv2.KalmanFilter(4, 2)

        # Constant-velocity transition model
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Only position is measured
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise: larger velocity noise allows fast adaptation to bounces
        self.kalman.processNoiseCov = np.diag(
            [1.0, 1.0, 1.5, 1.5]
        ).astype(np.float32)

        # Measurement noise: lower = more trust in detections, less lag
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0

        # High initial uncertainty → fast convergence on first measurements
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 100.0

        self.kalman_initialized = False

    # Internal physics simulation
    def _simulate_all(self, x, y, dx, dy, paddles, steps=40):
        """Runs the full physics simulation; returns (positions, bounce_flags).

        positions    – list of (int x, int y) for every step including the start.
        bounce_flags – parallel bool list; True when a wall or paddle bounce
                       occurred on that step (always False for the start point).
        One step equals one real video frame.
        """
        pts    = [(int(x), int(y))]
        bounced = [False]          # start point is never a bounce
        r      = self.PUCK_RADIUS
        W, H   = self.TABLE_W, self.TABLE_H

        for _ in range(steps):
            dx *= self.FRICTION
            dy *= self.FRICTION
            x  += dx
            y  += dy

            if np.sqrt(dx**2 + dy**2) < 0.3:
                break

            bounce = False

            # Wall bounces
            if x <= r:
                x  = r + (r - x);               dx = -dx * self.WALL_RESTITUTION;  bounce = True
            elif x >= W - r:
                x  = (W - r) - (x - (W - r));   dx = -dx * self.WALL_RESTITUTION;  bounce = True
            if y <= r:
                y  = r + (r - y);               dy = -dy * self.WALL_RESTITUTION;  bounce = True
            elif y >= H - r:
                y  = (H - r) - (y - (H - r));   dy = -dy * self.WALL_RESTITUTION;  bounce = True

            # Paddle bounces
            for ux, uy, ur in paddles:
                dist = np.sqrt((x - ux)**2 + (y - uy)**2)
                if 0 < dist < r + ur:
                    nx, ny = (x - ux) / dist, (y - uy) / dist
                    dot    = dx * nx + dy * ny
                    if dot < 0:
                        dx  = (dx - 2*dot*nx) * self.PADDLE_RESTITUTION
                        dy  = (dy - 2*dot*ny) * self.PADDLE_RESTITUTION
                        overlap = (r + ur) - dist
                        x  += nx * overlap
                        y  += ny * overlap
                        bounce = True

            pts.append((int(x), int(y)))
            bounced.append(bounce)

        return pts, bounced

    # Main pipeline
    def preprocess_frame(self, frame):
        """Applies perspective warp and produces red-color and combined motion masks."""
        warped_frame  = cv2.warpPerspective(frame, self.matrix, (self.TABLE_W, self.TABLE_H))
        blurred_frame = cv2.GaussianBlur(warped_frame, (11, 11), 0)
        hsv_frame     = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        # Motion mask via background subtraction
        fg_mask = self.backSub.apply(blurred_frame)

        # Red color mask (covers both paddles and puck)
        lower_red1, upper_red1 = np.array([0,   120, 60]), np.array([10,  255, 255])
        lower_red2, upper_red2 = np.array([170, 120, 60]), np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv_frame, lower_red1, upper_red1),
            cv2.inRange(hsv_frame, lower_red2, upper_red2),
        )

        #Combined mask: moving red objects only (puck candidates)
        kernel       = np.ones((5, 5), np.uint8)
        combined_mask = cv2.bitwise_and(red_mask, fg_mask)
        combined_mask = cv2.erode( combined_mask, kernel, iterations=1)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

        return warped_frame, red_mask, combined_mask

    def detect_objects(self, red_mask, combined_mask):
        """Detects paddles (large red circles) and puck candidates (small moving red circles)."""
        valid_candidates = []
        paddles          = []

        #Paddles: large red contours
        for contour in cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(contour)
            if not (self.PADDLE_AREA_MIN < area < self.PADDLE_AREA_MAX):
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            if 4 * np.pi * area / (perimeter**2) > self.PADDLE_CIRCULARITY:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    paddles.append((cx, cy, int(np.sqrt(area / np.pi))))

        #Puck candidates: small, circular, moving, not overlapping a paddle
        for contour in cv2.findContours(combined_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(contour)
            if not (self.PUCK_AREA_MIN < area < self.PUCK_AREA_MAX):
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            if 4 * np.pi * area / (perimeter**2) > self.PUCK_CIRCULARITY:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    too_close = any(
                        np.sqrt((cx - px)**2 + (cy - py)**2) < pr + 20
                        for px, py, pr in paddles
                    )
                    if not too_close:
                        valid_candidates.append({
                            "center":      (cx, cy),
                            "circularity": 4 * np.pi * area / (perimeter**2),
                        })

        return valid_candidates, paddles

    def track_puck(self, valid_candidates):
        """Runs one Kalman filter step and returns the current puck position estimate."""
        center = None

        if not self.kalman_initialized:
            #Initialization phase: require N consecutive detections
            if valid_candidates:
                best = max(valid_candidates, key=lambda c: c["circularity"])
                cx, cy = best["center"]

                # Reset buffer if detection jumps too far (likely noise)
                if self.init_buffer:
                    prev_x, prev_y = self.init_buffer[-1]
                    if np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2) > 80:
                        self.init_buffer = []

                self.init_buffer.append((cx, cy))
                center = (cx, cy)

                if len(self.init_buffer) >= self.INIT_CONFIRMATIONS:
                    avg_x   = float(np.mean([p[0] for p in self.init_buffer]))
                    avg_y   = float(np.mean([p[1] for p in self.init_buffer]))
                    init_vx = float(self.init_buffer[-1][0] - self.init_buffer[-2][0])
                    init_vy = float(self.init_buffer[-1][1] - self.init_buffer[-2][1])

                    self.kalman.statePost = np.array(
                        [[avg_x], [avg_y], [init_vx], [init_vy]], dtype=np.float32
                    )
                    self.kalman_initialized = True
                    self.init_buffer        = []
                    self.stable_frames      = 0
                    center                  = (int(avg_x), int(avg_y))
            else:
                self.init_buffer = []  # broken sequence → restart

        else:
            #Tracking phase: predict → associate → correct
            predicted       = self.kalman.predict().flatten()
            pred_x, pred_y  = int(predicted[0]), int(predicted[1])

            if valid_candidates:
                best = min(
                    valid_candidates,
                    key=lambda c: np.sqrt(
                        (c["center"][0] - pred_x)**2 +
                        (c["center"][1] - pred_y)**2
                    ),
                )
                dist = np.sqrt(
                    (best["center"][0] - pred_x)**2 +
                    (best["center"][1] - pred_y)**2
                )

                if dist < 200:
                    # Good match: update filter with measurement
                    measurement = np.array(
                        [[float(best["center"][0])],
                         [float(best["center"][1])]],
                        dtype=np.float32,
                    )
                    corrected      = self.kalman.correct(measurement).flatten()
                    center         = (int(corrected[0]), int(corrected[1]))
                    self.lost_frames   = 0
                    self.stable_frames = min(self.stable_frames + 1, self.MIN_STABLE_FRAMES)
                else:
                    # Suspicious detection: trust prediction instead
                    center             = (pred_x, pred_y)
                    self.lost_frames  += 1
            else:
                # Occlusion: no detection, use Kalman prediction
                center             = (pred_x, pred_y)
                self.lost_frames  += 1
                self.stable_frames = max(0, self.stable_frames - 1)

            # Full reset after too many consecutive lost frames
            if self.lost_frames >= self.MAX_LOST_FRAMES:
                self.pts.clear()
                self._init_kalman()
                self.init_buffer   = []
                self.lost_frames   = 0
                self.stable_frames = 0
                self.smooth_dx     = 0.0
                self.smooth_dy     = 0.0
                center             = None

        self.pts.appendleft(center)
        if center is not None:
            self.pos_history.append((self.frame_counter, center))
        return center

    def predict_trajectory(self, paddles):
        """Simulates the puck trajectory forward and returns waypoints for drawing."""
        pred_pts = []

        # Wait until Kalman is stable
        if not self.kalman_initialized or self.stable_frames < self.MIN_STABLE_FRAMES:
            return pred_pts
        if self.pts[0] is None:
            return pred_pts

        # Starting position: current Kalman-filtered position
        norm_x = float(self.pts[0][0])
        norm_y = float(self.pts[0][1])

        # Velocity from Kalman state + EMA smoother + shock filter for bounces
        kv     = self.kalman.statePost.flatten()
        raw_dx = float(kv[2])
        raw_dy = float(kv[3])

        # Shock filter: instant reset on direction reversal (bounce), EMA otherwise
        if self.smooth_dx * raw_dx < 0:
            self.smooth_dx = raw_dx
        else:
            self.smooth_dx = 0.75 * self.smooth_dx + 0.25 * raw_dx

        if self.smooth_dy * raw_dy < 0:
            self.smooth_dy = raw_dy
        else:
            self.smooth_dy = 0.75 * self.smooth_dy + 0.25 * raw_dy

        norm_dx = self.smooth_dx
        norm_dy = self.smooth_dy

        if abs(norm_dx) < 1.0 and abs(norm_dy) < 1.0:
            return pred_pts  # puck is nearly stationary

        # Run simulation once: reuse results for both evaluation and visualization
        all_pts, bounce_flags = self._simulate_all(
            norm_x, norm_y, norm_dx, norm_dy, paddles
        )
        self.pred_history.append((self.frame_counter, all_pts))

        # Build waypoint list: start + every bounce point + end
        pred_pts.append(all_pts[0])
        for i in range(1, len(all_pts) - 1):
            if bounce_flags[i]:
                pred_pts.append(all_pts[i])
        if len(all_pts) > 1:
            pred_pts.append(all_pts[-1])
        return pred_pts

    # Evaluation
    def get_error_stats(self, horizons=(5, 10, 20)):
        """Returns rolling prediction error statistics for the last ~60 frames.

        For each horizon H: finds predictions made exactly H frames ago and
        compares the predicted position at step H with the actual tracked position.
        Each (prediction_frame, horizon) pair is counted once in accumulated_errors.

        Returns: {H: {'mean': px, 'std': px, 'count': n}} or None if no data.
        """
        pos_dict = dict(self.pos_history)
        result   = {}

        for h in horizons:
            errors = []
            for pred_frame, sim_pts in self.pred_history:
                target = pred_frame + h
                if target in pos_dict and h < len(sim_pts):
                    px, py = sim_pts[h]
                    ax, ay = pos_dict[target]
                    err = float(np.sqrt((px - ax)**2 + (py - ay)**2))
                    errors.append(err)

                    # Accumulate for lifetime stats, each pair counted once
                    pair = (pred_frame, h)
                    if pair not in self._evaluated_pairs:
                        self._evaluated_pairs.add(pair)
                        if h in self.accumulated_errors:
                            self.accumulated_errors[h].append(err)

            result[h] = {
                'mean':  float(np.mean(errors)),
                'std':   float(np.std(errors)),
                'count': len(errors),
            } if errors else None

        return result

    def get_accumulated_stats(self):
        """Returns prediction error statistics aggregated over the entire video."""
        result = {}
        for h, errors in self.accumulated_errors.items():
            result[h] = {
                'mean':  float(np.mean(errors)),
                'std':   float(np.std(errors)),
                'count': len(errors),
            } if errors else None
        return result

    # Visualisation
    def draw_elements(self, warped_frame, center, paddles, pred_pts):
        """Draws paddles, puck, history trail, and predicted trajectory."""
        # Paddles
        for ux, uy, ur in paddles:
            cv2.circle(warped_frame, (ux, uy), ur, (255, 0, 255), 2)

        # Puck
        if center is not None:
            cv2.circle(warped_frame, center, self.PUCK_RADIUS, (0, 255, 0), 3)
            cv2.circle(warped_frame, center, 3,               (0,   0, 255), -1)

        # Position history trail (fades with distance)
        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            thickness = int(np.sqrt(20 / float(i + 1)) * 2.5)
            cv2.line(warped_frame, self.pts[i - 1], self.pts[i], (0, 255, 255), thickness)

        # Predicted trajectory
        if len(pred_pts) > 1:
            for j in range(1, len(pred_pts)):
                cv2.line(warped_frame, pred_pts[j - 1], pred_pts[j], (0, 0, 255), 3)
            cv2.circle(warped_frame, pred_pts[-1], self.PUCK_RADIUS, (0, 165, 255), 2)