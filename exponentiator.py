class bernoulli_cubature:
    def __init__(self, degree : int):
        """Creates a new Bernoulli cubature object from the given integer degree."""

        import warnings
        import uuid
        self.id = uuid.uuid4()
        self.degree = degree

    def __hash__(self):
        return hash(self.id)

    def eval(self, power : int):
        """Evaluates Bernoulli moment from the given integer power."""
        power = power[0]
        if power > self.degree:
            raise ValueError("Insufficient cubature degree.")
        elif power < 0:
            raise ValueError("Non-negative cubature power required.")
        else:
            return int((1/2)*(1+(-1)**power))

def _gaussian_moments(degree : int):
    """Helper function to generate Gaussian moments up to given integer degree."""
    moments = [1,0]
    for i in range(2,degree+1):
        moments.append(((i-1)*moments[-2]))
    return moments

class gaussian_cubature:
    def __init__(self, degree):
        """Creates a new Gaussian cubature object from the given integer degree."""
        import uuid
        import warnings

        self.id = uuid.uuid4()
        self.degree = degree
        self.gaussian_moments = _gaussian_moments(degree)

    def __hash__(self):
        return hash(self.id)

    def eval(self, power):
        """Evaluates Gaussian moment from the given integer power."""
        power = power[0]
        if power > self.degree:
            raise ValueError("Insufficient cubature degree.")
        elif power < 0:
            raise ValueError("Non-negative cubature power required.")
        else:
            return self.gaussian_moments[power]
        
class correlated_gaussian_cubature:
    def __init__(self,degree):
        assert degree <= 7

        import uuid
        import warnings

        self.id = uuid.uuid4()
        self.degree = degree
        self.moments = [[1,0,1,0,3,0,15,0],[0,np.sqrt(1/3),0,1,0,1,0],[1,0,5/3,0,1,0],[0,1,0,1,0],[3,0,1,0],[0,1,0],[15,0],[0]]

    def eval(self,power):
        power1, power2 = power
        if power1 + power2 > self.degree:
            raise ValueError("Insufficient cubature degree.")
        elif power1 < 0 or power2 < 0:
            raise ValueError("Non-negative cubature power required.")
        else:
            return self.moments[power1][power2]
        
class numeric:
    def __init__(self, value):
        """Creates a new numerical object to interact with cubature - with given constant value."""
        self.value = value

    def eval(self, power : int):
        """Evaluates constant raised to given integer power."""
        power = power[0]
        return self.value**power
    
from functools import total_ordering

@total_ordering
class basis_term:
    def __init__(self, word : str):
        """Creates a basis term object (using the Eulerian idomptent spanning set) from the given word."""
        self.index = word

    def __len__(self):
        return len(self.index)+self.index.count("0")

    def __int__(self):
        return int(self.index)

    def __lt__(self, other):
        if isinstance(other, basis_term):
            return self.index < other.index
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, basis_term):
            return self.index == other.index
        return NotImplemented
    
    def __hash__(self):
        return hash(self.index)
    
    def __str__(self):
        return "e("+str(self.index)+")"
    
from collections import defaultdict

class TupleCounter:
    def __init__(self):
        """Creates an helper object that adapts Counter (from collections) to include counting elementwise in tuples."""
        self.counter = defaultdict(lambda: (0, 0))  # Default value is a tuple (0, 0)

    def update(self, iterable):
        """Adds iterable to tuple counter."""
        for key, value in iterable:
            if key in self.counter:
                # Combine the tuples by summing their respective elements
                self.counter[key] = tuple(map(sum, zip(self.counter[key], value)))
            else:
                self.counter[key] = value

    def __getitem__(self, key):
        return self.counter[key]

    def __setitem__(self, key, value):
        self.counter[key] = value

    def __repr__(self):
        return f'{self.counter}'
    
class coefficient:
    def __init__(self, terms:list):
        self.terms = terms # sample: [{gauss : 4, 1 : 1, A : 4.5,...}, {-5 : 1, gauss : 1, bernoulli : 1,...}]

    def __mul__(self, other):
        if isinstance(other, coefficient):
            #from collections import Counter
            new_terms = []
            for x in self.terms:
                for y in other.terms:
                    tc = TupleCounter()
                    tc.update(x.items())
                    tc.update(y.items())
                    new_terms.append(dict(tc.counter))
            return coefficient(new_terms)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __str__(self):
        string = ""
        for x in self.terms:
            for y in x.items():
                if isinstance(y[0],gaussian_cubature):
                    string += "(G^" + str(y[1])+")"
                elif isinstance(y[0],bernoulli_cubature):
                    string += "(B^" + str(y[1])+")"
                elif isinstance(y[0],numeric):
                    string += "(" + str(y[0].value) + "^" + str(y[1])+")"
            string += "+"
        return string[:-1]
    
    def __repr__(self):
        return self.__str__() 


    def eval(self):
        if len(self.terms) == 1:
            run_tot = 1
            term_dict = self.terms[0]
            term_vals = list(term_dict.values())
            for i, x in enumerate(term_dict.keys()):
                if isinstance(x, (numeric, bernoulli_cubature, gaussian_cubature, correlated_gaussian_cubature)):
                    run_tot = x.eval(term_vals[i])*run_tot
                else:
                    raise ValueError("Unsupported value type in coefficient terms.")
            return run_tot
        else:
            return sum([coefficient([x]).eval() for x in self.terms])
        
class symmetrised_product:
    def __init__(self,items:list):
        self.items = sorted(items, key = lambda x : (len(x), int(x)))

    def __len__(self):
        length = 0
        for item in self.items:
            length += len(item)
        return length
    
    def __eq__(self,other):
        return self.items == other.items
    
    def __hash__(self):
        return hash(frozenset(self.items))

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        if len(self.items) == 0:
            return "1"
        string = ""
        for item in self.items:
            string += str(item)+","
        return "("+string[:-1]+")" 
    
    def num_permutations(self):
        from itertools import permutations
        return len(set(permutations(self.items)))
    
from math import factorial

def exponentiate(structure:list, degree:int):
    results = [{symmetrised_product([]):coefficient([{numeric(1) : (1,)}])}]
    for i in range(degree):
        new_level = {}
        for x in results[-1].items():
            for y in structure:
                sym = symmetrised_product(x[0].items+[y[0]])
                if len(sym) <= degree and sym not in new_level.keys():
                    coeff = x[1]*y[1]
                    new_level.update({sym:coeff})
        results.append(new_level)

    translate_results = []
    for index, val in enumerate(results):
        new_level = {}
        for x,y in val.items():
            evaluated = y.eval()
            if evaluated != 0: 
                new_level.update({str(x) : evaluated*x.num_permutations()/factorial(index)})
        translate_results.append(new_level)
    return(translate_results)