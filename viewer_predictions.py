import sys
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtCore import Qt
from simple_transformer_model import SimpleTransformerModel
from train_sample_transformer import SimpleNuScenesDataset

class PredictionViewer(QGraphicsView):
    def __init__(self, ground_truth, predictions):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.scale_factor = 1.0  # 초기 확대/축소 비율

        self.draw_trajectories(ground_truth, predictions)
        self.setWindowTitle("Prediction Viewer")
        self.resize(800, 600)
        self.show()

    def draw_trajectories(self, ground_truth, predictions):
        pen_gt = QPen(QColor("blue"))
        pen_gt.setWidth(2)
        pen_pred = QPen(QColor("red"))
        pen_pred.setWidth(2)

        # 데이터의 범위 설정
        scale_factor = 10  # 데이터를 화면 크기에 맞게 스케일 조정
        offset_x, offset_y = 400, 300  # 화면 중심을 기준으로 이동

        for gt, pred in zip(ground_truth, predictions):
            for i in range(len(gt) - 1):
                self.scene.addLine(
                    gt[i, 0] * scale_factor + offset_x,
                    -gt[i, 1] * scale_factor + offset_y,
                    gt[i + 1, 0] * scale_factor + offset_x,
                    -gt[i + 1, 1] * scale_factor + offset_y,
                    pen_gt,
                )
            for i in range(len(pred) - 1):
                self.scene.addLine(
                    pred[i, 0] * scale_factor + offset_x,
                    -pred[i, 1] * scale_factor + offset_y,
                    pred[i + 1, 0] * scale_factor + offset_x,
                    -pred[i + 1, 1] * scale_factor + offset_y,
                    pen_pred,
                )

    def wheelEvent(self, event):
        """마우스 휠로 확대/축소"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            # 확대
            self.scale(zoom_in_factor, zoom_in_factor)
            self.scale_factor *= zoom_in_factor
        else:
            # 축소
            self.scale(zoom_out_factor, zoom_out_factor)
            self.scale_factor *= zoom_out_factor

    def mousePressEvent(self, event):
        """마우스 드래그 시작"""
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """마우스 드래그 종료"""
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

def load_data():
    # Load dataset
    DATA_PATH = "./converted_data/converted_data.npy"
    dataset = SimpleNuScenesDataset(DATA_PATH)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    # Load trained model
    model = SimpleTransformerModel(
        input_dim=2,
        hidden_dim=64,
        num_heads=4,
        num_modes=3,
        future_len=30
    ).cuda()
    model.load_state_dict(torch.load("./saved_simple_transformer_model.pth"))
    model.eval()

    ground_truth = []
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            ego_in = batch["ego_in"].cuda()
            ground_truth_batch = batch["ground_truth"].cpu().numpy()

            _, trajectories = model(ego_in)
            best_predictions = trajectories[:, 0, :, :].cpu().numpy()

            ground_truth.extend(ground_truth_batch)
            predictions.extend(best_predictions)
    return np.array(ground_truth), np.array(predictions)

def main():
    app = QApplication(sys.argv)
    ground_truth, predictions = load_data()
    print(ground_truth)
    print(predictions)
    #viewer = PredictionViewer(ground_truth, predictions)
    #viewer.show()
    #sys.exit(app.exec_())

if __name__ == "__main__":
    main()