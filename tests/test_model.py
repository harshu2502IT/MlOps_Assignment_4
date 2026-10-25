import os
from preprocess import preprocess
from train import train


def test_preprocess_creates_files(tmp_path):
    out = tmp_path / "processed"
    preprocess("data/data.csv", str(out))
    assert (out / "train.csv").exists()
    assert (out / "test.csv").exists()


def test_train_accuracy(tmp_path):
    proc, models = tmp_path / "p", tmp_path / "m"
    preprocess("data/data.csv", str(proc))
    acc = train(str(proc), str(models))
    assert acc > 0.8
    assert os.path.exists(models / "model.joblib")
