from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "cases"
    tasks = {
        "algorithms" : ["simple_classification"],
        "datasets": ["indian_pines"],
        "target_sizes" : [32]
    }
    ev = TaskRunner(tasks,1,tag,verbose=True)
    ev.evaluate()