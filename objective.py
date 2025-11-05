def objective(trial):
    
    # optimizer:
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128 ]) # usual candidates 
    #epochs = trial.suggest_categorical("epochs", [10, 20, 50, 100, 200, 500, ])
    epochs = 3
    # optim = trial.suggest_categorical("optim", ["Adam", "SGD"]) 
    
    # model itself 

    conv1 = trial.suggest_categorical("conv1", [32, 64, 128 ])
    conv2 = trial.suggest_categorical("conv2", [32, 64, 128 ]) 
    fc1 = trial.suggest_categorical("fc1",  [128, 256, 512 , 1024]) 

    # data loader 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN(conv1 = conv1, conv2 = conv2, fc1=fc1, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    with mlflow.start_run() as run:
        mlflow.log_param("optimizer", "adam") # 
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("conv1", conv1)
        mlflow.log_param("conv2", conv2)
        mlflow.log_param("fc1", fc1)

        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                optimizer.zero_grad() # gradient set to 0
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Gradient descent
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0) 
            
            accuracy = correct / total 
            mlflow.log_metric("loss", running_loss, step = epoch)
            mlflow.log_metric("train_accuracy", accuracy, step = epoch)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Train accuracy: {accuracy:.4f}")
        
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                predicted = torch.max(outputs.data, 1)[-1] # argmax von outputs
                total += labels.size(0) # Gesamte Anzahl der Labels
                correct += (predicted==labels).sum().item() # Anzahl der richtig klassifizierten Labels
        
        accuracy = correct / total
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.pytorch.log_model(model, "model" , input_example=input_example)
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)

    return accuracy 
