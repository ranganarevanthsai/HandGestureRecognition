import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model


@dataclass
class PredictResult:
    symbol: str
    skeleton_bgr: Optional[np.ndarray] = None


class ASLPredictor:
    """
    Backend-friendly predictor extracted from final_pred.py.
    Produces the same 400x400 landmark skeleton image and applies the same
    landmark rule logic to split the 8 CNN groups into A-Z (plus control tokens).
    """

    def __init__(self, model_path: str = "cnn8grps_rad1_model.h5", offset: int = 29):
        self.model = load_model(model_path)
        self.offset = offset
        self.hd = HandDetector(maxHands=1)
        self.hd2 = HandDetector(maxHands=1)
        self.pts = None

    @staticmethod
    def _distance(x, y) -> float:
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def _draw_skeleton_400(self, pts, w: int, h: int) -> np.ndarray:
        white = np.zeros((400, 400, 3), dtype=np.uint8)
        white[:] = 255

        os = ((400 - w) // 2) - 15
        os1 = ((400 - h) // 2) - 15

        for t in range(0, 4):
            cv2.line(
                white,
                (pts[t][0] + os, pts[t][1] + os1),
                (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                (0, 255, 0),
                3,
            )
        for t in range(5, 8):
            cv2.line(
                white,
                (pts[t][0] + os, pts[t][1] + os1),
                (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                (0, 255, 0),
                3,
            )
        for t in range(9, 12):
            cv2.line(
                white,
                (pts[t][0] + os, pts[t][1] + os1),
                (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                (0, 255, 0),
                3,
            )
        for t in range(13, 16):
            cv2.line(
                white,
                (pts[t][0] + os, pts[t][1] + os1),
                (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                (0, 255, 0),
                3,
            )
        for t in range(17, 20):
            cv2.line(
                white,
                (pts[t][0] + os, pts[t][1] + os1),
                (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                (0, 255, 0),
                3,
            )

        cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
        cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
        cv2.line(
            white,
            (pts[13][0] + os, pts[13][1] + os1),
            (pts[17][0] + os, pts[17][1] + os1),
            (0, 255, 0),
            3,
        )
        cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
        cv2.line(
            white,
            (pts[0][0] + os, pts[0][1] + os1),
            (pts[17][0] + os, pts[17][1] + os1),
            (0, 255, 0),
            3,
        )

        for i in range(21):
            cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

        return white

    def _cnn_group_predict(self, skeleton_bgr_400: np.ndarray) -> Tuple[int, int]:
        white = skeleton_bgr_400.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white, verbose=0)[0], dtype="float32")
        ch1 = int(np.argmax(prob, axis=0))
        prob[ch1] = 0
        ch2 = int(np.argmax(prob, axis=0))
        return ch1, ch2

    def _postprocess_symbol(self, ch1: int, ch2: int) -> str:
        # The rule logic below is adapted from final_pred.py (Application.predict).
        # It expects self.pts (21 hand landmarks) to be set.
        pts = self.pts
        pl = [ch1, ch2]

        l = [
            [5, 2],
            [5, 3],
            [3, 5],
            [3, 6],
            [3, 0],
            [3, 2],
            [6, 4],
            [6, 1],
            [6, 2],
            [6, 6],
            [6, 7],
            [6, 0],
            [6, 5],
            [4, 1],
            [1, 0],
            [1, 1],
            [6, 3],
            [1, 6],
            [5, 6],
            [5, 1],
            [4, 5],
            [1, 4],
            [1, 5],
            [2, 0],
            [2, 6],
            [4, 6],
            [1, 0],
            [5, 7],
            [1, 6],
            [6, 1],
            [7, 6],
            [2, 5],
            [7, 1],
            [5, 4],
            [7, 0],
            [7, 5],
            [7, 2],
        ]
        if pl in l:
            if pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]:
                ch1 = 0

        l = [[2, 2], [2, 1]]
        if pl in l and (pts[5][0] < pts[4][0]):
            ch1 = 0

        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (
                pts[0][0] > pts[8][0]
                and pts[0][0] > pts[4][0]
                and pts[0][0] > pts[12][0]
                and pts[0][0] > pts[16][0]
                and pts[0][0] > pts[20][0]
                and pts[5][0] > pts[4][0]
            ):
                ch1 = 2

        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self._distance(pts[8], pts[16]) < 52:
                ch1 = 2

        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (
                pts[6][1] > pts[8][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]
                and pts[0][0] < pts[8][0]
                and pts[0][0] < pts[12][0]
                and pts[0][0] < pts[16][0]
                and pts[0][0] < pts[20][0]
            ):
                ch1 = 3

        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l and (pts[4][0] > pts[0][0]):
            ch1 = 3

        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l and (pts[2][1] + 15 < pts[16][1]):
            ch1 = 3

        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l and (self._distance(pts[4], pts[11]) > 55):
            ch1 = 4

        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self._distance(pts[4], pts[11]) > 50 and (
                pts[6][1] > pts[8][1]
                and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]
            ):
                ch1 = 4

        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l and (pts[4][0] < pts[0][0]):
            ch1 = 4

        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l and (pts[1][0] < pts[12][0]):
            ch1 = 4

        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (
                pts[6][1] > pts[8][1]
                and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]
                and pts[4][1] > pts[10][1]
            ):
                ch1 = 5

        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
                ch1 = 5

        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l and (pts[4][0] > pts[0][0]):
            ch1 = 5

        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
                ch1 = 5

        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l and (pts[3][0] < pts[0][0]):
            ch1 = 7

        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l and (pts[6][1] < pts[8][1]):
            ch1 = 7

        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l and (pts[18][1] > pts[20][1]):
            ch1 = 7

        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l and (pts[5][0] > pts[16][0]):
            ch1 = 6

        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
                ch1 = 6

        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l and (self._distance(pts[8], pts[16]) > 50):
            ch1 = 6

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l and (self._distance(pts[4], pts[11]) < 60):
            ch1 = 6

        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l and (pts[5][0] - pts[4][0] - 15 > 0):
            ch1 = 6

        l = [
            [5, 0],
            [5, 1],
            [5, 4],
            [5, 5],
            [5, 6],
            [6, 1],
            [7, 6],
            [0, 2],
            [7, 1],
            [7, 4],
            [6, 6],
            [7, 2],
            [5, 0],
            [6, 3],
            [6, 4],
            [7, 5],
            [7, 2],
        ]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = 1

        l = [
            [6, 1],
            [6, 0],
            [0, 3],
            [6, 4],
            [2, 2],
            [0, 6],
            [6, 2],
            [7, 6],
            [4, 6],
            [4, 1],
            [4, 2],
            [0, 2],
            [7, 1],
            [7, 4],
            [6, 6],
            [7, 2],
            [7, 5],
            [7, 2],
        ]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = 1

        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if (
                pts[6][1] > pts[8][1]
                and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]
                and pts[2][0] < pts[0][0]
                and pts[4][1] > pts[14][1]
            ):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self._distance(pts[4], pts[11]) < 50 and (
                pts[6][1] > pts[8][1]
                and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]
            ):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if (
                pts[6][1] > pts[8][1]
                and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]
                and pts[2][0] < pts[0][0]
                and pts[14][1] < pts[4][1]
            ):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l and (pts[5][0] - pts[4][0] - 15 < 0):
            ch1 = 1

        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = 1

        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[4][0] < pts[5][0] + 15) and (
                pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]
            ):
                ch1 = 7

        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[4][1] > pts[14][1]:
                ch1 = 1

        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (
                pts[0][0] + fg < pts[8][0]
                and pts[0][0] + fg < pts[12][0]
                and pts[0][0] + fg < pts[16][0]
                and pts[0][0] + fg < pts[20][0]
            ) and not (
                pts[0][0] > pts[8][0]
                and pts[0][0] > pts[12][0]
                and pts[0][0] > pts[16][0]
                and pts[0][0] > pts[20][0]
            ) and self._distance(pts[4], pts[11]) < 50:
                ch1 = 1

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]):
            ch1 = 1

        # map groups to letters
        if ch1 == 0:
            ch1 = "S"
            if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                ch1 = "A"
            if (
                pts[4][0] > pts[6][0]
                and pts[4][0] < pts[10][0]
                and pts[4][0] < pts[14][0]
                and pts[4][0] < pts[18][0]
                and pts[4][1] < pts[14][1]
                and pts[4][1] < pts[18][1]
            ):
                ch1 = "T"
            if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                ch1 = "E"
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
                ch1 = "M"
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                ch1 = "N"

        if ch1 == 2:
            ch1 = "C" if self._distance(pts[12], pts[4]) > 42 else "O"

        if ch1 == 3:
            ch1 = "G" if self._distance(pts[8], pts[12]) > 72 else "H"

        if ch1 == 7:
            ch1 = "Y" if self._distance(pts[8], pts[4]) > 42 else "J"

        if ch1 == 4:
            ch1 = "L"

        if ch1 == 6:
            ch1 = "X"

        if ch1 == 5:
            if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                ch1 = "Z" if pts[8][1] < pts[5][1] else "Q"
            else:
                ch1 = "P"

        if ch1 == 1:
            if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = "B"
            if pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]:
                ch1 = "D"
            if pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = "F"
            if pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = "I"
            if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]:
                ch1 = "W"
            if (
                pts[6][1] > pts[8][1]
                and pts[10][1] > pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]
                and pts[4][1] < pts[9][1]
            ):
                ch1 = "K"
            if (self._distance(pts[8], pts[12]) - self._distance(pts[6], pts[10])) < 8 and (
                pts[6][1] > pts[8][1]
                and pts[10][1] > pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]
            ):
                ch1 = "U"
            if (self._distance(pts[8], pts[12]) - self._distance(pts[6], pts[10])) >= 8 and (
                pts[6][1] > pts[8][1]
                and pts[10][1] > pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]
                and pts[4][1] > pts[9][1]
            ):
                ch1 = "V"
            if (pts[8][0] > pts[12][0]) and (
                pts[6][1] > pts[8][1]
                and pts[10][1] > pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]
            ):
                ch1 = "R"

        # Control tokens used by original app for typing/backspace
        if ch1 in (1, "E", "S", "X", "Y", "B"):
            if pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = " "

        if ch1 in ("E", "Y", "B"):
            if (pts[4][0] < pts[5][0]) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]
            ):
                ch1 = "next"

        if ch1 in ("next", "B", "C", "H", "F", "X"):
            if (
                pts[0][0] > pts[8][0]
                and pts[0][0] > pts[12][0]
                and pts[0][0] > pts[16][0]
                and pts[0][0] > pts[20][0]
                and pts[4][1] < pts[8][1]
                and pts[4][1] < pts[12][1]
                and pts[4][1] < pts[16][1]
                and pts[4][1] < pts[20][1]
                and pts[4][1] < pts[6][1]
                and pts[4][1] < pts[10][1]
                and pts[4][1] < pts[14][1]
                and pts[4][1] < pts[18][1]
            ):
                ch1 = "Backspace"

        return str(ch1)

    def predict_from_bgr(self, frame_bgr: np.ndarray) -> PredictResult:
        """
        Takes a BGR frame (e.g., decoded from browser upload) and returns the
        predicted symbol and the 400x400 skeleton image used for the CNN.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return PredictResult(symbol="")

        # Mirror like the desktop app.
        frame_bgr = cv2.flip(frame_bgr, 1)

        hands = self.hd.findHands(frame_bgr, draw=False, flipType=True)
        if not hands or not hands[0]:
            return PredictResult(symbol="", skeleton_bgr=None)

        hand = hands[0][0]
        x, y, w, h = hand["bbox"]

        y1 = max(0, y - self.offset)
        y2 = min(frame_bgr.shape[0], y + h + self.offset)
        x1 = max(0, x - self.offset)
        x2 = min(frame_bgr.shape[1], x + w + self.offset)
        cropped = frame_bgr[y1:y2, x1:x2]
        if cropped.size == 0:
            return PredictResult(symbol="", skeleton_bgr=None)

        handz = self.hd2.findHands(cropped, draw=False, flipType=True)
        if not handz or not handz[0]:
            return PredictResult(symbol="", skeleton_bgr=None)

        hand2 = handz[0][0]
        self.pts = hand2["lmList"]

        skeleton = self._draw_skeleton_400(self.pts, w=w, h=h)
        ch1, ch2 = self._cnn_group_predict(skeleton)
        symbol = self._postprocess_symbol(ch1, ch2)
        return PredictResult(symbol=symbol, skeleton_bgr=skeleton)

