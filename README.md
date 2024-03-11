# Ringier_task

Follow the instructions below to build the proper environment for this project, train the model on the training dataset,and use the trained model to make predictions on the test data (`predict_payload.json`). You should have conda package manager installed to be able to successfully run the commands below.

1. Place the following files in the data folder:
    ```
    - train_data.json
    - predict_payload.json
    - taxonomy_mappings.json
    ```
2. In the terminal, run the following command to build the project environment.
    ```
    source env.sh
    ```
3. After running the above command, the environment of the project will be activated in your current shell. Run the command below to start training the model.
    ```
    python train.py
    ```
4. When the training is finished, a new folder called `models` will be created that contains the saved model. Now you can make predictions on the `predict_payload.json` data using the following command:
    ```
    python predict.py --path_to_payload data/predict_payload.json
    ```
    The predictions will be saved in a newly created folder called `predictions/`.
    ```
    predictions/probas.json
    ```
5. You can also use the following API deployed on the Internet to test the solution. To do so, run the following lines in Python. For this part, you will not need the environment of the project necessarily. Any python environment with `requests` and `json` packages installed should work. You just need to substitute the placeholder `<PATH-TO-predict_payload.json>` with the appropriate path on your computer.
    ```
    import requests
    import json
    headers = {'Content-Type': 'application/json', 'Accept':'application/json'}

    payload = {
        "data": json.load(open("<PATH-TO-predict_payload.json>", "r"))
    }
    response = requests.post("https://ringier-c45baz7vfq-oa.a.run.app", json=payload, headers=headers)
    # The predicted probabilities for each class:
    print(response.json())
    ```
