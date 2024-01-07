# Homework 2 - Classifying with Missing or Noisy Labels

This homework consists of 2 tasks.

### Task 1
Learning to classify with missing labels. This is a classic practical example since in most cases, we have access to data similar to what we want to optimize for, but lack the ground-truth labels.

The task involves learning to classify while having only a fraction of the class labels for each category. To simulate real-world conditions, there will be a slight imbalance in class percentages between annotated and unannotated samples. Some classes will have a higher percentage of missing labels than others. It's important to note that this class distribution might not reflect the distribution at inference time.

### Task 2
Learning to classify with noisy labels. This scenario often occurs when using non-specialized annotators or labels generated from an automated system above a minimal acceptance threshold.

To simulate this, we understand that there is a limit on the total percentage of available data. If we “know” that this mechanism produces a wrong ratio of α, then we can assume (1−α) is correctly labeled. However, this assumption may not apply universally to all classes. Some classes might be easily separable for this annotation mechanism. Given that some categories are harder to distinguish, certain label pairs may have more noisy annotations.

As a result, some labels might become either under-represented or over-represented in the label space.

### Important Note
For Task 2, I changed the column names to match the format used in the first task within the code.

### How to run
The code is structured to execute the best results obtained from my experiments. To run either of the two tasks, you need to specify the root directory where Task1 and Task2 are located. To execute a specific task, it is necessary to specify the task name.
```
python main.py --dataset_root "/Users/marianluca/Documents/Master/ATAI_DNN" --task_name "task1" --n_epoch 1
```
