import preprocessing
import model
import evaluation
import visualization

base_dir = 'Data'
preprocessor = preprocessing.RetinaDatasetPreprocessor(base_dir)

model = model.Model(preprocessor.train, preprocessor.train_mask, preprocessor.test, preprocessor.test_mask)
model.train_model()

eval = evaluation.Eval(preprocessor, model)
eval.data_size()
eval.evaluate_model()
eval.generate_metrics_table()


viz = visualization.Viz(preprocessor, model)
viz.display_images_with_masks()
viz.display_images_with_predictions()
viz.plot_precision_recall_curve()
viz.plot_iou_scores()
viz.display_error_maps()
viz.display_feature_maps()
viz.plot_confusion_matrix()
viz.plot_loss_curve()
viz.display_best_worst()

