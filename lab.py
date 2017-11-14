"""6.009 Lab 8A: carlae Interpreter"""

import sys


class EvaluationError(Exception):
    """Exception to be raised if there is an error during evaluation."""
    pass


class Function:
    def __init__(self, parent, function):
        """
            intialize values for function
        """
        self.params = {}
        self.env = parent
        self.function = function

    def start_params(self, params):
        """
            assing the paramters to blank variables in the function
        """
        for p in params:
            self.params[p] = None
        self.params['parent'] = self.env

    # def assign_params(self, values):
    #     index = 0
    #     for p in self.params:
    #         self.params[p] = values[index]
    #         index +=1
    def get_params(self):
        return list(self.params.keys())

    def __str__(self):
        # print('here i am')
        return 'function to do ' + str(self.function)

    # def evaluate(self, values):
    #     """
    #         function to evaluate the function
    #     """
    #     ## first go and assign the parameters 
    #     for index, val in values:
    #         if type(val) == int or type(val) == float:
    #             self.params[index] = val
    #         if type(val) == str:
    #             try:
    #                 self.params[index] = get_env(self.env, val)[val]
    #             except:
    #                 raise EvaluationError
    #     return self.function

class Environment:
    def __init__(self, parent, env):
        self.env = env
        self.parent = parent
    def access_env(self):
        return self.env
    def check_val(self, val):
        raise NotImplementedError

def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a carlae
                      expression
    """
    
    # return tokenize_helper(source)
    tokenized_list = []
    for word in tokenize_helper(source):
        tokenized_list.append(word)
    return tokenized_list

def tokenize_helper(source):
    """ helpher for tokenize
    """
    pos = 0
    index = 0
    while index < len(source):
        char = source[index]
        # print(index, source[pos:index], char)
        if char == '(':
            yield char
            pos = index + 1
        if char == ' ':
            if source[pos:index] != '' and source[pos:index] != ' ':
                yield source[pos:index]
            pos = index + 1
        if char == ')':
            if pos != 0 and pos != index:
                yield source[pos:index]
            yield char
            pos = index + 1
        if index < len(source) - 1:
            if source[index:index+1] == '\n':
                    yield source[pos:index]
                    index +=1 
                    pos = index
        if char == ';':
            try:
                pos = index = index +  source[index:].index('\n') + 2
            except:
                pos = index = len(source)
        index += 1
    # print(pos, len(source))
    if pos != len(source):
        yield source[pos:]

# example = ";add the numbers2 and 3\n ( + ; this expression\n 2     ; spans multiple\n 3  ; lines\n\n)"
# print(tokenize(example))

def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    # raise NotImplementedError
    parsed = parse_helper(tokens)
    # print('mine prints', parsed[0])
    return parsed[0]

def parse_helper(tokens):
    # print(tokens)
    if tokens == []:
        return tokens
    if tokens[0] == '(':
        paren = 1
        for r_index, val in enumerate(tokens[1:]):
            if val == '(':
                paren+=1
            if val == ')':
                paren-=1
            if paren == 0:
                break
        if paren != 0:
            raise SyntaxError
        r_index += 1
        return [parse_helper(tokens[1:r_index])] + parse_helper(tokens[r_index+1:])
    if tokens[0] == ')':
        raise SyntaxError
    if not convert_int(tokens[0]):
        # print('hello')
        if not convert_float(tokens[0]):
            return [tokens[0]] + parse_helper(tokens[1:])
        return [convert_float(tokens[0])] + parse_helper(tokens[1:])
    else:
        return [convert_int(tokens[0])] + parse_helper(tokens[1:])

def convert_int(x):
    try:
        # f = float(x)
        return int(x)
    except:
        return False

def convert_float(x):
    try:
        return float(x)
    except:
        return False

def multiply(l):
    a = 1
    for val in l:
        a = a*val
    return a

def divide(l):
    if len(l) == 2:
        return l[0]/l[1]
    else:
        return l[0]/multiply(l[1:])

carlae_builtins = {
    '+': sum,
    '-': lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    '*': multiply,
    '/': divide,
}


def get_env(env, symbol):
    new_env = env.copy()
    while new_env != None:
        # if not env.get(symbol):
        #     env = env['parent']
        # else:
        #     return env.get(symbol)
        if symbol in new_env:
            return new_env
        else:
            new_env = new_env['parent']
    return None

def evaluate(tree, env = None):
    """
    Evaluate the given syntax tree according to the rules of the carlae
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    # raise NotImplementedError

    if env == None:
        env = {'parent': carlae_builtins}

    return evaluate_helper(tree, env)


def evaluate_helper(tree, env = None):
    # print(tree, env)
    if type(tree) == int or type(tree) == float:
        return tree
    elif type(tree) == str:
        try:
            if tree not in env:
                env = get_env(env, tree)
            return env[tree]
        except:
            raise EvaluationError
    elif type(tree) == list:
        if len(tree) == 1:
            return tree[0]
        keyword, evl = tree[0], tree[1:]
        # print(keyword, evl)
        # if type(evl) == list:
        #     # val = 0
        #     # for e in evl:
        #     #     print(e)
        #     #     val += evaluate_helper(e, env)
        #     # print(val)
        #     # key, e = evl[0][0], evl[0][1:][0]
        #     # print(key, e)
        #     # val = evaluate_helper([key, e], env)
        #     # print(val)
        #     for e in evl[0]:
        #         print('first')
        #         print(e)
        #         # print(e)
        #         x = evaluate_helper(e, env)

        if type(keyword) == list:
            # print(keyword)
            f = evaluate_helper(keyword, env)
            # print([f, evl])
            # print(env)
            # print(f.params)
            # val = f(evl)
            # print(val)
            val = evaluate_helper([f, evl], env)
            # print(val)
            return val

        elif type(keyword) == Function:
            function = keyword
            orig_env = function.env.copy()
            # print('this is a function call')
            params = function.get_params()
            # print(function, params, evl[0])
            evl = evl[0]
            # print(params, evl)
            index = 0
            for p in params:
                if p == 'parent':
                    continue
                # print(p, evl[index], type(evl[index]))
                if type(evl[index]) == str:
                    function.params[p] = get_env(orig_env, evl[index])[evl[index]]
                    index +=1
                if type(evl[index]) == list:
                    function.params[p] = evaluate_helper(evl[index], env)
                    index +=1
                if type(evl[index]) == int or type(evl[index]) == float:
                    function.params[p] = evl[index]
                    index +=1
            var = evaluate_helper(function.function, function.params)
            # print(var)
            # print(function.params, function.function)
            return var

        elif keyword == 'define':
            # val = evaluate_helper(tree, env)
            # env[keyword] = val
            # print('evl', evl)
            var, ev = evl[0], evl[1]
            # print('ev', ev)
            val = evaluate_helper(ev, env)
            # print('the val is', val)
            env[var] = val
            return val

        elif keyword == 'lambda':
            params, function = evl
            # print(params, function)
            function = Function(env, function)
            function.start_params(params)
            # print('created function')
            # print(function)
            return function
        
        elif keyword not in env:
            # print(env)
            orig_env = env.copy()
            work_env = get_env(env, keyword)
            # print('working', work_env)
            # print('original', orig_env)
            if not work_env:
                raise EvaluationError
            # return evaluate_helper(tree, env)
            for index, var in enumerate(evl):
                if type(var) == str:
                    # print(var, 'is not an int')
                    evl[index] = get_env(orig_env, var)[var]
                    # env = orig_env
                if type(var) == list:
                    evl[index] = evaluate(var, env)
            # print(evl)
            return work_env[keyword](evl)

        elif keyword in env:
            if type(env[keyword]) == Function:
                function = env[keyword]
                orig_env = function.env.copy()
                # print('this is a function call')
                params = function.get_params()
                # print(params, evl)
                index = 0
                for p in params:
                    if p == 'parent':
                        continue
                    # print(p, evl[index], type(evl[index]))
                    if type(evl[index]) == str:
                        function.params[p] = get_env(orig_env, evl[index])[evl[index]]
                        index +=1
                    if type(evl[index]) == list:
                        function.params[p] = evaluate_helper(evl[index], env)
                        index +=1
                    if type(evl[index]) == int or type(evl[index]) == float:
                        function.params[p] = evl[index]
                        index +=1
                var = evaluate_helper(function.function, function.params)
                # print(var)
                # print(function.params, function.function)
                return var
            else:
                orig_env = env.copy()
                for index, var in enumerate(evl):
                    if type(var) == str:
                        evl[index] = get_env(orig_env, var)[var]
                        # env = orig_env
                    if type(var) == list:
                        evl[index] = evaluate_helper(var, env)
                return env[keyword](evl)

    # print(env)

def evaluate_helper_1(tree):
    if type(tree) == int or type(tree) == float:
        return tree
    elif type(tree) == str:
        if tree in carlae_builtins:
            return tree
        else:
            raise EvaluationError
    else:
        if len(tree) == 1:
            return tree
        return carlae_builtins[tree[0]](tree[1:])

def_env = {'parent': carlae_builtins}

def make_new_env(env = None):
    if not env:
        def_env = {'parent': carlae_builtins}
        return def_env
    return env

def result_and_env(tree, env = None):
    if env == None:
        env = {'parent': carlae_builtins}
    result = evaluate(tree, env)
    # print(result)
    return (result, env)

def REPL():
    inp = input('inpt > ')
    while inp != 'QUIT':
        try:
            env = make_new_env()
            # print('  outpt > ', result_and_env(parse(tokenize(inp)), def_env))
            print('  outpt > ', evaluate(parse(tokenize(inp)), def_env))
        except:
            e = sys.exc_info()[0]
            print('  outpt: ', e)
        # print(evaluate(parse(tokenize(inp))))
        inp = input('inpt > ')

if __name__ == '__main__':
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)
    # pass
    REPL()
