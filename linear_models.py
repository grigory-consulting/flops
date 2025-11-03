X = 2 * np.random.rand(100,1) 
y = 4 + 3 * X + np.random.randn(100,1) 
plt.plot(X,y, "b.")
plt.xlabel("x")
plt.ylabel("y")
plt.axis([0,2,0,15])
plt.show()









n_epochs = 50 
batch_size = 10
m = X.shape[0]
eta = 0.01 
w_mbgd = np.random.randn(2,1) 

for epoch in range(n_epochs): 
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices] 
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, batch_size):
        xi = X_b_shuffled[i:i+batch_size]
        yi = y_shuffled[i:i + batch_size]
        gradients = 2/batch_size * xi.T @ (xi @ w_mbgd - yi) 
        w_mbgd = w_mbgd - eta*gradients                      
w_mbgd
