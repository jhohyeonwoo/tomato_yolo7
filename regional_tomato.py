
from utils.plots import plot_results

if __name__ == '__main__':
    # 여기에 학습 코드가 들어간다고 가정
    # 예: python train.py --weights yolov7.pt --data data.yaml ...

    # 학습 결과 시각화 자동 저장
    plot_results(save_dir='runs/train/exp')
