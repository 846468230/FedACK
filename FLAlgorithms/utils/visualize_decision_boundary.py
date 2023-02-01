import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression,SGDClassifier
import torch
import os
def visualize_decision_boundary(server):
    features = []
    labels = []
    server.model.to_device()
    test_datas = server.dataset.load_data_i_client("test")
    for i, batch in enumerate(test_datas):
        with torch.no_grad():
            feature = server.model(batch, server.lingual_model, e_reps=True)['feature']
            if len(features) == 0:
                features = feature.cpu().numpy()
                labels = batch.labels
            else:
                features = np.concatenate((features, feature.cpu().numpy()), axis=0)
                labels = np.concatenate((labels, batch.labels), axis=0)
    visualize_it(features, labels, server.model, server.dataset.cur_dataset, -1, server.dataset.noniid_alpha, server.algorithm,
                 server.dataset.base_path)
    server.model.move_from_device()
    for user in server.users:
        features = []
        labels= []
        user.model.to_device()
        user.student_model.to_device()
        test_datas = user.datas.load_data_i_client("test")
        for i, batch in enumerate(test_datas):
            with torch.no_grad():
                feature = user.model(batch,user.cross_lingual_model,e_reps=True)['feature']
                if len(features)==0:
                    features = feature.cpu().numpy()
                    labels = batch.labels
                else:
                    features = np.concatenate((features,feature.cpu().numpy()),axis=0)
                    labels = np.concatenate((labels,batch.labels),axis=0)
        visualize_it(features,labels,user.model,user.datas.cur_dataset,user.id,user.datas.noniid_alpha,user.algorithm,user.datas.base_path,user.student_model)
        user.model.move_from_device()
        user.student_model.move_from_device()

def visualize_it(features,labels,model,dataset_name,client_id,noniid_alpha,algorithm,base_path,model2=None):
    plt.close("all")
    sns.set_style("white")
    # features, labels = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)
    min1, max1 = features[:, 0].min(), features[:, 0].max()
    min1,max1 = min1-(max1-min1)*0.1, max1+(max1-min1)*0.1
    min2, max2 = features[:, 1].min(), features[:, 1].max()
    min2,max2 = min2-(max2-min2)*0.1, max2+(max2-min2)*0.1
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = torch.from_numpy(np.hstack((r1, r2)).astype(np.float32)).cuda()
    yhat = model.forward_to_classify(grid)['output'].argmax(dim=1)
    zz = yhat.cpu().detach().numpy().reshape(xx.shape)
    plt.contour(xx, yy, zz, [1, ], cmap='Paired')
    # grid = np.hstack((r1, r2))
    # model = LogisticRegression()
    # # fit the model
    # model.fit(features, labels)
    # model2 = SGDClassifier()
    # # fit the model
    # model2.fit(features, labels)
    # # make predictions for the grid
    # yhat = model.predict(grid)
    # zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    if model2 !=None:
        yhat = model2.forward_to_classify(grid)['output'].argmax(dim=1)
        zz = yhat.cpu().detach().numpy().reshape(xx.shape)
        plt.contour(xx, yy, zz, [1, ], cmap='Paired')
    # yhat = model2.predict(grid)
    # zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    X_df = pd.DataFrame(features, columns=['x1', 'x2'])
    y_df = pd.DataFrame(labels, columns=["class"])
    frames = [X_df, y_df]
    data = pd.concat(frames, axis=1)
    g1 = sns.scatterplot(x="x1", y="x2",hue="class",style="class", data=data)
    # g1.legend(set_title=None)
    # # create scatter plot for samples from each class
    # for class_value in range(2):
    #     # get row indexes for samples with this class
    #     row_ix = np.where(labels == class_value)
    #     # create scatter of these samples
    #     plt.scatter(features[row_ix, 0], features[row_ix, 1], cmap='Paired')
    filepath = os.path.join(base_path, "figures")
    # if not os.path.exists(filepath):
    #     os.makedirs(filepath)
    filepath = os.path.join(filepath, f"{algorithm}client{client_id}{dataset_name}A{noniid_alpha}.pdf")
    plt.savefig(filepath)
    plt.show()

