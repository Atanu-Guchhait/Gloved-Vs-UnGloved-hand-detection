# Part 2: Reasoning-Based Questions

## Q1: Choosing the Right Approach

I would use object detection because the task is not just to say whether a label exists, but to locate where the label should be on the product and verify its presence. Detection allows the model to focus on the specific region where the label is expected, rather than learning the entire product appearance. Classification may fail because the products are visually very similar except for a small label area. Segmentation would be unnecessarily complex unless the exact shape or area of the label matters. If detection does not work well, my fallback would be binary classification on a cropped region of interest, where the model only sees the label area.

---

## Q2: Debugging a Poorly Performing Model

First, I would compare training images with real factory images to check for differences in lighting, camera angle, background, or resolution. Next, I would visualize predictions on validation images to see if the model is making systematic mistakes, such as confusing background patterns. I would also review annotation quality to ensure labels are accurate and consistent. Then, I would test performance on small controlled subsets, such as only well-lit images or only one product type. Finally, I would try fine-tuning the model with a small set of newly collected factory images to reduce domain shift.

---

## Q3: Accuracy vs Real Risk

Accuracy is not the right metric in this case because missing defective products has a much higher cost than flagging extra safe ones. I would focus more on recall, especially for the defective class, since catching every defective item is critical. Precision is less important if false alarms can be manually checked. I would also analyze the confusion matrix to understand exactly what types of errors the model is making. In a real production system, minimizing false negatives matters more than having a high overall accuracy score.

---

## Q4: Annotation Edge Cases

Blurry or partially visible objects should usually be kept in the dataset because they represent real-world conditions the model will face. Removing them may artificially inflate performance during training but cause failures during deployment. However, they should be labeled carefully and consistently to avoid confusing the model. The trade-off is between realism and noise: too many poor-quality samples can hurt learning, but removing all of them reduces robustness. A balanced approach is to keep them but ensure they do not dominate the dataset.
