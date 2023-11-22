import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def pca(X,k):#k is the components you want
  #mean of each feature
  n_samples, n_features = X.shape
  mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
  #normalization
  norm_X=X-mean
  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)
  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  plt.scatter(X[:, 0], X[:, 1])
  # 画y=o的直线,平行于x轴；
  plt.axhline(y=0, color='r', lw=2, label='分割线')
  # 画x=4的直线，平行于y轴；
  plt.axvline(x=0, color='r', lw=2, label='分割线2')
  plt.grid(linestyle=":", color="grey")
  f=feature.tolist()
  plt.quiver(0, 0, f[0][0]*100, f[0][1]*100, angles='xy', scale_units='xy', scale=1)
  f.append([0,0])
  res=[[0,f[0][0]],[0,f[0][1]]]
  # print(f)
  plt.plot(feature)
  plt.show()
  print(res)
  #get new data
  data=np.dot(norm_X,np.transpose(feature))
  return data


if __name__ == '__main__':
    data=np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #自写的方法结果：
    res=pca(data,k=1)
    print('自写方法的结果:')
    print(res)
    #Sklearn结果
    model=PCA(n_components=1)
    res=model.fit(data)
    print('Sklearn方法结果:')
    print(res.transform(data))
