PJ_Of_Face_Recognition
====
1.Description
----
      Based on tf1.3.0, it's developed for unmanned supermarket & subway turnstiles to improve performance.
      I use The Basel Face Model 2009 to generate training dataset, so that I can train the neural network
      without having to take many photos.To generate photos in more circumstances, matlab is used to illuminate
      faces from different angles. For more detail, you can view this website:http://gravis.dmi.unibas.ch/PMM/.
2.Improvement
----
      180314:Migrated from my pj of pulsar detection, I try to input the 3D Basel dataset to train its NN.
      
      180325:I successfully input the standford data into the classifier, unluckily the overfitting 
      phenomenon was unacceptable...
