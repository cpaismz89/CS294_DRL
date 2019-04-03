# Importations
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
import os
# DRL BC model (class)
class DRL_BCModel_Free(object):

    # Constructor
    def __init__(self, _IData, _OData, _Instance, _Model, _nNet=[], _NLayers=2, 
                 _batch_size=64, _activation=[tf.nn.tanh, tf.nn.tanh]):
        # Main values for the model (Data)
        self.IData = _IData
        self.OData = np.reshape(_OData, (_OData.shape[0], _OData.shape[2]))
        
        # Batch size to be used during the training
        self.batch_size = _batch_size
        
        # Model parameters
        self.nNet = _nNet
        self.NLayers = _NLayers
        
        # Instance (Humanoid, Hopper,...) name and model (BC, EXPERT, DAgger)
        self.Instance = _Instance
        self.ModelName = _Model
        
        # New session of tensorflow 
        self.sess = tf.Session()
        
        # Activation rules
        self.activation = _activation
        
        # Model
        self.MPath = os.getcwd() + "/Models/"
        if not os.path.exists(self.MPath):
            os.makedirs(self.MPath)
        
        # Shape of data (for TF layers)
        self.IShape = [self.batch_size, self.IData.shape[-1]]
        self.OShape = [self.batch_size, self.OData.shape[-1]]
        
        # Variables / Placeholders for TF
        self.IPlaceHolder = tf.placeholder(tf.float32, shape=self.IShape)
        self.OPlaceHolder = tf.placeholder(tf.float32, shape=self.OShape)
        
        # Create the model calling the method inside the class
        if len(self.nNet) == 0:
            self.Model = self.createModel(self.IPlaceHolder, nNet = [self.batch_size, 32], 
                                          NLayers=2, activation=self.activation)
        else:
            self.Model = self.createModel(self.IPlaceHolder, self.nNet, self.NLayers, activation=self.activation)
        
        # Loss function: 
        # From TF DOC: Computes half the L2 norm of a tensor without the sqrt -> output = sum(t ** 2) / 2
        self.LossF = tf.reduce_mean(tf.nn.l2_loss(self.OPlaceHolder - self.Model))
        self.LossS = tf.summary.scalar("LossF", self.LossF)
        
    # Print-out information (Class object) 
    def printOut(self):
        print("Instance:", self.Instance)
        print("Model Name:", self.ModelName)
        print("Batch Size:", self.batch_size)
        print("Input Shape:", self.IShape)
        print("Output Shape:", self.OShape)
        print("IPlaceholder:", self.IPlaceHolder)
        print("OPlaceholder:", self.OPlaceHolder)
        print("Model:", self.Model)
        print("LossF:", self.LossF)
        
    # Create the RLN model based on the generated placeholder and activation rules
    def createModel(self, IPlaceHolder, nNet, NLayers, activation):
        RLNModel = tf.layers.dense(inputs=IPlaceHolder, units=nNet[0], activation=activation[0], use_bias=True)
        for layer in range(1,NLayers):
            RLNModel = tf.layers.dense(inputs=RLNModel, units=nNet[layer], activation=activation[layer], use_bias=True)
        RLNModel = tf.layers.dense(inputs=RLNModel, units=self.OShape[-1])
        return RLNModel
    
    # Train the model
    def train(self, epochs=100, TrainData = None, TestData = None, nIter=None):
        # If it is the first call to the model, create training and testing datasets based on random shuffling
        if TrainData is None and TestData == None:
            TrainData, TestData = shuffle(self.IData, self.OData, random_state=0)
            #print("Training and Testing data is generated")
            #print("TrainData:", TrainData)
            #print("TestData:", TestData)
        else:
            TestData = np.reshape(TestData, (TestData.shape[0], TestData.shape[2]))
            #print("TestData is reshaped:", TestData)
            
        # Create an optimizer (ADAM) with respect to the loss function
        opt = tf.train.AdamOptimizer(learning_rate=1e-4,
                                    beta1=0.9,
                                    beta2=0.999,
                                    epsilon=1e-08,
                                    use_locking=False,
                                    name='Adam').minimize(self.LossF)
        
        # Save the model
        modelSave = tf.train.Saver()
        
        # Batch IDxs
        batchIDxs =  len(TrainData) // self.batch_size
        
        # Summaries depending on the iteration
        if nIter is None:
            toWrite = tf.summary.FileWriter(self.MPath + self.Instance)
        else:
            toWrite = tf.summary.FileWriter(self.MPath + self.Instance + str(nIter))
        toWrite.add_graph(self.sess.graph)
            
        # Initialize the session
        self.sess.run(tf.global_variables_initializer())
        
        # Main training by number of epochs
        for niter in tqdm(range(epochs)):
            for index in range(batchIDxs):
                # Get batches for Train and Test data
                TrainSet = TrainData[index * self.batch_size : (index + 1) * self.batch_size]
                TestSet = TestData[index * self.batch_size : (index + 1) * self.batch_size]
                
                # Dictionary to feed the session when running
                toFeed = {self.IPlaceHolder:TrainSet, self.OPlaceHolder:TestSet}
                
                # Run the model
                self.sess.run(opt, feed_dict=toFeed)

                # Check-Points
                if index % 50 == 0:
                    valLoss = self.sess.run(self.LossS, feed_dict=toFeed)
                    toWrite.add_summary(valLoss, niter * batchIDxs + index)
                    
        # Save results to folder BC
        modelSave.save(self.sess, self.MPath + self.Instance + "/BCModel")

        
    # Sampling function
    def Sampling(self, Inputs, nNet = 64):
        toFeed = {self.IPlaceHolder: np.repeat(Inputs[None,:], nNet, axis=0)}
        Output = self.sess.run(self.Model, feed_dict=toFeed)
        return Output[0]