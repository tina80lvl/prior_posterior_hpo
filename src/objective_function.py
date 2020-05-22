from sklearn.metrics import f1_score

class ML(object):
   def __init__(self, model, X_train, y_train, X_val, y_val):
      self.model = model
      self.X_train = X_train
      self.y_train = y_train
      self.X_val = X_val
      self.y_val = y_val

   def __call__(self, hypers):
      model = self.model(kernel=hypers[0], prior=hypers[1],
                 noise=hypers[2], use_gradients=hypers[3],
                 normalize_output=hypers[4], ...)

      mean, _ = model.predict(X_val)
      y_pred = np.around(mean)
      print(y_pred)
      # calculating F-score
      f_score = f1_score(self.y_val, y_pred, average='macro')
      logger.debug("F-score: " + str(f_score))
      return model.score(x_val, y_val)
