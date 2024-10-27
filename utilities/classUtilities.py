class GenericReprBase:
    
    def get_info(self,delim = '\n '):
        name = type(self).__name__
        
        vars_list = []
        for key, value in vars(self).items():
            if isinstance(value, np.ndarray):
                value = value.flatten()
            vars_list.append(f'{key}={value!r}')
            
        vars_str = delim.join(vars_list)
        return f'{name}({vars_str})'

    def __repr__(self,delim = '\n '):
        return self.get_info(delim)