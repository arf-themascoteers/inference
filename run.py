from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "kan"
    tasks = {
        "algorithms" : ["kan_classification"],
        "datasets": ["indian_pines"],
        "target_sizes" : [10,200]
    }
    ev = TaskRunner(tasks,1,tag,verbose=True)
    ev.evaluate()