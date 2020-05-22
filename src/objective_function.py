class ML(object):
   def init(self, model_class, x_train, y_train, x_val, y_val):
      self.model_class = model_class
      self.x_train = x_train
      self.y_train = y_train
      self.x_val = x_val
      self.y_val = y_val

   def call(self, hyper_params):
      model = self.model_class(kernel=hyper_params[0], prior=hyper_params[1],
                 noise=hyper_params[2], use_gradients=hyper_params[3],
                 normalize_output=hyper_params[4], ...)
      model.fit(x_train, y_train)
      return model.score(x_val, y_val)

objective_function = ML(model_class, x_train, y_train, x_val, y_val)
