from app.game.backened.train import TrainPipeline

if __name__ == "__main__":
    model_path = None
    training_pipeline = TrainPipeline(model_path, is_shown = False) # shown仅控制训练时是否可视化   平谷时一定可视化
    training_pipeline.run_from_database()

