"""6.009 Lab 8A: carlae Interpreter"""

import sys


class EvaluationError(Exception):
    """Exception to be raised if there is an error during evaluation."""
    pass


class Function:
    def __init__(self, params, parent, function):
        """
        intialize values for function
            params: list of parameters, initially variables
            parent: parent pointer to parent env
            function: actual function to run
        """
        self.params = params
        self.env = parent
        self.function = function
    def __str__(self):
        """
        string representation of function
        """
        return 'function to do ' + str(self.function)

class Environment:
    def __init__(self, parent):
        """
        intialize values for environment:
            env: current environment var-->val
            parent: parent pointer to parent env
        """
        self.env = {}
        self.parent = parent
    def add_val(self, var, val):
        """
        add a value and variable to environment
        """
        self.env[var] = val
    def __str__(self):
        """
        string representation of environment
        """
        print('environemnt as following')
        for var in self.env:
            print(var, '->', self.env[var])

def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a carlae
                      expression
    """
    # intialize tokenize list
    tokenized_list = []
    # for every word returned by tokenize, append to list
    for word in tokenize_helper(source):
        tokenized_list.append(word)
    # return list
    return tokenized_list

def tokenize_helper(source):
    """ 
    helper for tokenize: iteratively return values
    """
    # intialize position to start from and index
    pos = 0
    index = 0
    # while index < string length
    while index < len(source):
        # get the value at this index
        char = source[index]
        # print(index, source[pos:index], char)
        # if this is a '(' we know to return this
        # increment the position as the next place to start from
        if char == '(':
            yield char
            pos = index + 1
        # if this is a space, return the stuff up to this space as long
        # as that is not a space
        # increment position as next place to start from
        if char == ' ':
            if source[pos:index] != '' and source[pos:index] != ' ':
                yield source[pos:index]
            pos = index + 1
        # if right paren, return stuff up to this
        # increment position
        if char == ')':
            if pos != 0 and pos != index:
                yield source[pos:index]
            yield char
            pos = index + 1
        # if index is not the second to last, check for newlines
        if index < len(source) - 1:
            # if newline exists, return up to this
            # increment index and position
            if source[index:index+1] == '\n':
                    yield source[pos:index]
                    index +=1 
                    pos = index
        # if comment (;), find the next newline or the end of the 
        # statement and change position and index to that value
        if char == ';':
            try:
                pos = index = index +  source[index:].index('\n') + 2
            except:
                pos = index = len(source)
        # increment index
        index += 1
    # print(pos, len(source))
    # if not at the end, return up to the end
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
    # get parsed values from helper and return
    # first index
    parsed = parse_helper(tokens)
    # print('mine prints', parsed[0])
    return parsed[0]

def parse_helper(tokens):
    """
    helper for parser: recursive function for tokens
    """
    # print(tokens)
    # if at the end of tokens, return empty string
    if tokens == []:
        return tokens
    # if the first value is a left parent, check for matching right paren
    if tokens[0] == '(':
        # start parent counter
        paren = 1
        # for the index and val, if it is
        #   left paren: add 1
        #   right parent: subtract 1
        # counter is 0, break and return index
        # if there is no matching, raise Error
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
        # recursively call helper on this subproblem and the rest of the tokens
        return [parse_helper(tokens[1:r_index])] + parse_helper(tokens[r_index+1:])
    # if there's a random unaccounted ), raise Error
    if tokens[0] == ')':
        raise SyntaxError
    # for non parens: return current value + parser on rest of tokens
    if not convert_int(tokens[0]):
        # print('hello')
        # then try to convert to float
        if not convert_float(tokens[0]):
            # return string representation if you can't do either
            return [tokens[0]] + parse_helper(tokens[1:])
        # if float possible, do float
        return [convert_float(tokens[0])] + parse_helper(tokens[1:])
    else:
        # if int possible, do int
        return [convert_int(tokens[0])] + parse_helper(tokens[1:])

def convert_int(x):
    """
    helper to try int conversion
    """
    try:
        # f = float(x)
        return int(x)
    except:
        return False

def convert_float(x):
    """
    helper to try float conversion
    """
    try:
        return float(x)
    except:
        return False

def multiply(l):
    """
    helper for multiply
    """
    a = 1
    # multiply vals in list
    for val in l:
        a = a*val
    return a

def divide(l):
    """
    helper for divide
    """
    # if two, divide 2
    if len(l) == 2:
        return l[0]/l[1]
    # if more than 2, divide first by rest
    else:
        return l[0]/multiply(l[1:])

carlae_builtins = {
    '+': sum,
    '-': lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    '*': multiply,
    '/': divide,
}

def get_env_part_2(env, symbol):
    """
    helper to return value of symbol given environment
    """
    # to not mutate env
    new_env = env
    # while the env is not None
    while new_env != None:
        # if thes symbol exists in the env, return value
        if symbol in new_env.env:
            return new_env.env[symbol]
        # set env to its parent
        else:
            new_env = new_env.parent
    # if not found, raise Error
    raise EvaluationError


def evaluate_function(keyword, evl):
    """
    helper to evaluate a function given a keyword and the params
    to evaluate
    """
    # if this is a basic keyword in builtins, evaluate builtin
    if keyword in carlae_builtins.values():
        return keyword(evl)
    # if the parameter lengths do not match, raise Error
    elif len(keyword.params) != len(evl):
        raise EvaluationError
    # if this is not a function, raise Error
    elif type(keyword) != Function:
        raise EvaluationError
    else:
        # print(keyword, evl)
        # print('here')
        # create working environemnt
        this_env = Environment(keyword.env)
        # for every value to evaluate, define the parameter variable as this value
        for i in range(len(evl)):
            evaluate_helper(['define', keyword.params[i], evl[i]], this_env)
        # evaluate this function in the environment with updated parameters
        return evaluate_helper(keyword.function, this_env)

def evaluate_helper(tree, env = None):
    """ 
    recursive helper for evaluate
    """
    # if tree is simple int or float, return value
    if type(tree) == int or type(tree) == float:
        return tree
    # if tree is a string, try to return the value of the
    # var or raise an Exception if it does not exist
    elif type(tree) == str:
        try:
            return get_env_part_2(env, tree)
        except:
            raise EvaluationError
    # if tree is a function return function
    elif type(tree) == Function:
        return tree
    # if tree is a list, do things with components
    elif type(tree) == list:
        # if tree is empty, raise error
        if not tree:
            raise EvaluationError
        # grab the keyword, var, and evl values
        try:
            keyword, var, evl = tree[0], tree[1], tree[2]
        # if there aren't three, just grab keyword and rest
        except:
            keyword, rest = tree[0], tree[1:]
        # print(keyword, var, evl)
        # if this is define
        if keyword == 'define':
            # if the variable is a list (generally a function)
            if type(var) == list:
                # change the representation to a lambda function
                # representation so that it is easier to work with
                new_def = ['define', var[0], ['lambda', var[1:], evl]]
                # evaluate and get function
                val = evaluate_helper(new_def[2], env)
                # add this function 
                env.add_val(new_def[1], val)
                # env.env[new_def[1]] = val
                return val
            # if not, this is probably a variable
            else:
                # evaluate the variable
                val = evaluate_helper(evl, env)
                # add to environemnt
                env.add_val(var, val)
                # env.env[var] = val
                # print(env)
                return val
        # if this is a function
        elif keyword == 'lambda':
            # create a function with the env as parent, variables,
            # and the function to evaluate 
            function = Function(var, env, evl)
            return function
        # if this thing is none of the above, actually evaluate
        else:
            try:
                # grab the function of the keyword
                function = evaluate_helper(keyword, env)
                params = []
                # for every value in the rest which is a paren
                for val in tree[1:]:
                    # evaluate this val to get value
                    v = evaluate_helper(val, env)
                    # add this to the parameters
                    params.append(v)
                # evaluate this function with these parameters
                return evaluate_function(function, params)
            except:
                # raise Error if this doesn't work for some reason
                raise EvaluationError
    else:
        # raise Error if nothing works
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

    # create working env
    env = create_new_env(env)
    # run helper and get values
    return evaluate_helper(tree, env)

def create_new_env(env):
    # create original parent of all: builtins
    builtins = Environment(None)
    builtins.env = carlae_builtins
    # if parent doesn't exist, create new env with this as parent
    if not env:
        env = Environment(builtins)
    # return env
    return env

def result_and_env(tree, env = None):
    """
    function to return both the result and current environment
    """
    # create env from helper
    env = create_new_env(env)
    # get result from evaluate
    result = evaluate(tree, env)
    # return result and env
    return (result, env)

def REPL():
    """
    Read, Evaluate, Print Loop for testing code
    """
    # grab input
    inp = input('inpt > ')
    # while not quit
    env = create_new_env(None)
    while inp != 'QUIT':
        # try to evaluate input
        try:
            print('  outpt > ', evaluate(parse(tokenize(inp)), env))
        # catch exception, print
        except:
            e = sys.exc_info()[0]
            print('  outpt: ', e)
        # grab next output
        inp = input('inpt > ')

if __name__ == '__main__':
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)
    # pass
    REPL()
