"""6.009 Lab 8A: carlae Interpreter"""

import sys


class EvaluationError(Exception):
    """Exception to be raised if there is an error during evaluation."""
    pass


class Function:
    def __init__(self, params, parent, function):
        """
            intialize values for function
        """
        self.params = params
        # self.env = parent.copy()
        self.env = parent
        # self.env = parent
        self.function = function
    def __str__(self):
        # print('here i am')
        return 'function to do ' + str(self.function)

class Environment:
    def __init__(self, parent):
        self.env = {}
        self.parent = parent
    # def access_env(self):
    #     return self.env
    # def check_val(self, val):
    #     raise NotImplementedError
    def add_val(self, var, val):
        self.env[var] = val

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

def get_env_part_2(env, symbol):
    if symbol in env.env:
        return env.env[symbol]
    if env.parent:
        return get_env_part_2(env.parent, symbol)
    raise EvaluationError


def evaluate_function(keyword, evl):
    if keyword in carlae_builtins.values():
        return keyword(evl)
    elif len(keyword.params) != len(evl):
        raise EvaluationError
    elif type(keyword) != Function:
        raise EvaluationError
    else:
        # print('here')
        # print(keyword.env)
        this_env = Environment(keyword.env)
        for i in range(len(evl)):
            # evaluate_helper(['define', keyword.params[i], evl[i]], keyword.env)
            evaluate_helper(['define', keyword.params[i], evl[i]], this_env)
        # print(keyword.)
        # return evaluate_helper(keyword.function, keyword.env)
        return evaluate_helper(keyword.function, this_env)

def evaluate_helper(tree, env = None):
    # print(tree, env)
    if type(tree) == int or type(tree) == float:
        # print('here')
        return tree
    elif type(tree) == str:
        # print(int(tree))
        # print(tree)
        # try:
        #     return int(tree)
        # except:
        #     pass
        try:
            # if tree not in env:
            #     env = get_env(env, tree)
            # return env[tree]
            # print('here')
            # print(env)
            return get_env_part_2(env, tree)
        except:
            raise EvaluationError
    elif type(tree) == Function:
        return tree
    elif type(tree) == list:
        # if len(tree) == 1:
        #     return tree[0]
        if not tree:
            raise EvaluationError
        try:
            keyword, var, evl = tree[0], tree[1], tree[2]
        except:
            keyword, rest = tree[0], tree[1:]
        # print(keyword, var, evl)
        if keyword == 'define':
            # if type(var) == list:
            # var, evl = rest[0], rest[1]
            # print(var, evl)
            if type(var) == list:
                new_def = ['define', var[0], ['lambda', var[1:], evl]]
                # new_def = ['define', rest[0][1], rest[0][1:], rest[1]]
                # print(new_def)
                val = evaluate_helper(new_def[2], env)
                # print(val)
                # env[new_def[1]] = val
                env.add_val(new_def[1], val)
                # print(env)
                return val
            else:
                val = evaluate_helper(evl, env)
                # val = evaluate_helper(rest[1], env)
                # env[var] = val
                env.add_val(var, val)
                # print(env)
                return val
        elif keyword == 'lambda':
            # var, evl = rest[0], rest[1]
            function = Function(var, env, evl)
            return function
        else:
            # print('here')
            try:
                # print(keyword, env)
                function = evaluate_helper(keyword, env)
                # print(function)
                params = []
                # print(function)
                for val in tree[1:]:
                    # print('first', val)
                    v = evaluate_helper(val, env)
                    # print(v)
                    params.append(v)
                # print(params)
                # params = [evaluate_helper(i, env) for i in tree[1:]]
                # print(function, params)
                return evaluate_function(function, params)
            except:
                raise EvaluationError
    else:
        raise EvaluationError

def evaluate(tree, env = None):
    """
    Evaluate the given syntax tree according to the rules of the carlae
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    # raise NotImplementedError

    # if env == None:
    #     env = {'parent': carlae_builtins}
    builtins = Environment(None)
    # builtins.update_env(carlae_builtins)
    builtins.env = carlae_builtins
    if not env:
        env = Environment(builtins)

    return evaluate_helper(tree, env)


def evaluate_helper_2(tree, env = None):
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
        if len(tree) == 0:
            raise EvaluationError
        keyword, evl = tree[0], tree[1:]
        # print(keyword, evl)

        if type(keyword) == list:
            # print('here')
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
            # print(evl, params)
            if len(evl) != len(params[1:]):
                raise EvaluationError
            # print(params, evl)
            index = 0
            for p in params[1:]:
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
            # print('var', var, 'ev', ev)
            if type(var) == list:
                new_def = ['define', var[0], ['lambda', var[1:], ev]]
                # print(new_def)
                val = evaluate_helper(new_def, env)
                env[var[0]] = val
                # print(val)
                return val
            else:
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
            # print('here')
            # print(env)
            try:
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
            except:
                raise EvaluationError

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

# def_env = {'parent': carlae_builtins}

builtins = Environment(None)
# builtins.update_env(carlae_builtins)
builtins.env = carlae_builtins
def_env = Environment(builtins)

# def make_new_env(env = None):
#     if not env:
#         def_env = {'parent': carlae_builtins}
#         return def_env
#     return env

def result_and_env(tree, env = None):
    # if env == None:
    #     env = {'parent': carlae_builtins}
    # result = evaluate(tree, env)
    # # print(result)
    # return (result, env)
    
    # all_parent = Environment(None)
    # all_parent.update_env(carlae_builtins)
    # if not env:
    #     env = Environment(all_parent)
    # result = evaluate(tree, env)
    # return (result, env)
    builtins = Environment(None)
    # builtins.update_env(carlae_builtins)
    builtins.env = carlae_builtins
    if not env:
        env = Environment(builtins)
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
            # print(result_and_env(parse(tokenize(inp)), def_env))
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
