import numpy as np
from scipy.stats import norm


class GapFiller:
    def __init__(self, seed=89, cat_threshold=50, discrete_approximation={}):
        """
        Parameters
        ----------
        seed : int
            Random state for reproducibility
        cat_threshold : int
            Maximum number of unique values for a feature to be considered categorical
        discrete_approximation : dict
            A dictionary with column numbers and numbers of bins for discrete approximations.
            This can be applied to continuous distributions with nonnormal shapes.
        """

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.cat_threshold = cat_threshold
        self.cat_info = {}
        self.is_fit = False
        self.discrete_approximation = discrete_approximation


    def fit(self, X: 'ndarray'):
        """
        Estimates covariance matrix and it's Cholesky decomposition after
        processing categorical features.
        """

        self.X = X.copy()
        self.mu = np.zeros(X.shape[1])
        self.sigma = np.ones(X.shape[1])
        self.X = self.preprocess(self.X)
        self.is_fit = True

        self.cov = self.compute_cov(self.X)
        self.L = np.linalg.cholesky(self.cov)        

        
    def fill_new(self, X):
        """ ... """

        raise NotImplementedError
        

    def fit_fill(self, X):
        """
        Fits covariance matrix and fills missing values. The behavior is analagous
        to fit_predict method from sklearn unsupervised algorithms.
        """

        self.fit(X)

        for j, row in enumerate(self.X):
            if np.any(np.isnan(row)):
                self.X[j, :] = self.fill_one_row(row)
        
        self.X = self.inverse_preprocess(self.X)
        X_to_be_returned = X.copy()
        R = np.isnan(X)
        X_to_be_returned[R] = self.X[R]
        return X_to_be_returned
    

    def compute_cov(self, X):
        """Computes covariance matrix and remedies its possible negative definitness."""
        
        masked_for_cov = np.ma.array(X, mask=np.isnan(X))
        cov = np.array(
            np.ma.cov(masked_for_cov, rowvar=False)
        )
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        diagonal = np.diag(np.where(eigenvalues < 0, 1e-10, eigenvalues))
        cov_new = np.matmul(np.matmul(eigenvectors, diagonal), np.transpose(eigenvectors)) 
        return cov_new


    def preprocess(self, X):
        """
        Performs Khohlov transform to all columns with less than
        self.cat_threshold unique values. Other values are considered
        continuous and centered.
        """

        for j in range(X.shape[1]):  # итерируемся по столбцам
            unique_vals = np.unique(X[:, j])
            unique_vals = unique_vals[~np.isnan(unique_vals)]
            
            if len(unique_vals) <= self.cat_threshold:
                ind = ~np.isnan(X[:, j])  # данные без пропусков
                X[ind, j] = self.khohlov_transform(X[ind, j], j)
            elif j in self.discrete_approximation.keys():
                ind = ~np.isnan(X[:, j])
                X[ind, j] = self.discrete_transform(X[ind, j], j)
            else:
                if not self.is_fit:
                    self.mu[j] = np.nanmean(X[:, j])
                    self.sigma[j] = np.nanstd(X[:, j])
                X[:, j] -= self.mu[j]
                X[:, j] /= self.sigma[j]
                
        return X


    def inverse_preprocess(self, X):
        """Performs inverse Khohlov transform to categorical variables
        and adds means to continuous variables."""

        for j in range(X.shape[1]):
            if j in self.cat_info.keys():
                X[:, j] = self.inverse_khohlov_transform(X[:, j], j)
            elif j in self.discrete_approximation.keys():
                X[:, j] = self.inverse_discrete_transfrom(X[:, j], j)
            else:
                X[:, j] *= self.sigma[j]
                X[:, j] += self.mu[j]

        return X


    def khohlov_transform(self, x : 'ndarray', j: 'column index'):
        """
        Performs Khohlov transform to the column array x and appends
        self.cat_info with a dictionary for the inverse transform.
        
        x should not contain np.nan!
        """

        # создаем словарь порогов
        if not self.is_fit:
            cat_values, counts = np.unique(x, return_counts=True)
            quantiles = np.cumsum(counts / np.sum(counts))
            z_values = norm.ppf(quantiles)
            z_values[-1] = np.inf  # нужно явно прописать из-за численных приколов в norm.ppf
            self.cat_info[j] = [z_values, cat_values]

        # непосредственно преобразование в непрерывную величину
        noise = np.sort(self.rng.randn(len(x)))
        sorting_index = np.argsort(x)
        inverse_sorting_index = np.argsort(sorting_index)

        vals = np.unique(x)
        sorted_x = x[sorting_index]
        for v in vals:
            ind = sorted_x == v
            y = noise[ind]
            self.rng.shuffle(y)
            noise[ind] = y

        return noise[inverse_sorting_index]


    def inverse_khohlov_transform(self, x, j):
        """Performes inverse Khohlov transform to array. """

        thresholds, values = self.cat_info[j]
        
        @np.vectorize
        def inverse(x):
            return values[x < thresholds].min()

        return inverse(x)

    def discrete_transform(self, x, j):
        """ Transforms feature to perform discrete approximation. """

        if not self.is_fit:
            n_bins=self.discrete_approximation[j]
            counts, edges = np.histogram(x, bins=n_bins, density=False)
            quantiles = np.cumsum(counts / np.sum(counts))
            z_values = norm.ppf(quantiles)
            z_values[-1] = np.inf
            values = (edges[1:] + edges[:-1]) / 2
            self.discrete_approximation[j] = [z_values, values]

        noise = np.sort(self.rng.randn(len(x)))
        sorting_index = np.argsort(x)
        inverse_sorting_index = np.argsort(sorting_index)

        # тут можно не шафлить?
        return noise[inverse_sorting_index]


    def inverse_discrete_transfrom(self, x, j):
        """ Almost identical to inverse Khohlov transform. """

        thresholds, values = self.discrete_approximation[j]
        
        @np.vectorize
        def inverse(x):
            return values[x < thresholds].min()

        return inverse(x)

    
    def fill_one_row(self, x: 'ndarray with missing values'):
        """Fills one row."""

        sorting_index = np.argsort(x)
        inverse_sorting_index = np.argsort(sorting_index)  # если запихнуть этот индекс в отсортированный массив, он вернется в изначальном порядке
        
        x_to_be_filled = x[sorting_index].copy()  # т.е. просто np.sort(x); сортировка поставит наны в конец
        noise = np.zeros_like(x)

        for j, elem in enumerate(x_to_be_filled):
            if np.isnan(elem):
                noise[j] = self.rng.randn()
                x_to_be_filled[j] = self.L[j] @ noise
            else:
                noise[j] = (elem - self.L[j] @ noise) / self.L[j, j]  # тут elem == x_to_be_filled[j]
                
        return x_to_be_filled[inverse_sorting_index]