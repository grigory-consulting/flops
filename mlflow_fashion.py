mlflow.set_experiment("FASHION")
model = NN()
lr = 1e-4 # lerning rate
loss = nn.CrossEntropyLoss() # CE because multi-class problem 
# optimizer = optim.SGD(model.parameters(), lr = lr) # stochastic gradient descent
optimizer = optim.Adam(model.parameters(), lr = lr) # modern version of stochastic gradient descent
n_epochs = 1

with mlflow.start_run() as run:
    for epoch in range(n_epochs): # training loop
        model.train() # train mode 
        running_loss = 0.0 # loss per epoch 
        for images,labels in train_loader:
            optimizer.zero_grad() # reset gradient 
            # forward 
            outputs = model(images) # calculate outputs
            curr_loss = loss(outputs, labels) # loss between output and label 
            running_loss += curr_loss
            # backward 
            curr_loss.backward() # gradients
            optimizer.step() # Update the weights 
        print(f"Epoch [{epoch +1}/{n_epochs}], Loss: {running_loss}")
    # Evaluation 

    model.eval() # setting the model to evaluation mode (implementation optimization)
    correct = 0
    total = 0

    with torch.no_grad(): # we are not interested in gradient anymore 
        for images, labels in test_loader:
            outputs = model(images)
            predicted = torch.max(outputs.data, 1)[-1] 
            total += labels.size(0) 
            correct += (predicted==labels).sum().item()

    accuracy = correct/total
    print(f"Accuracy: {accuracy*100:.2f}%")
