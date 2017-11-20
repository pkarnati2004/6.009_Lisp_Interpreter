"""6.009 Lab 8B: carlae Interpreter Part 2"""

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

class LinkedList:
    def __init__(self, elt):
        """
        initialize values for LinkedList:
            elt: current value
            next: pointer to next element
        """
        self.elt = elt
        self.next = None
    def __add__(self, other):
        """
        function to specify concatenation of lists
        """
        # if this is None or the value is none, return the list other
        if self is None or self.elt is None:
            return other
        # if other is None or other value is none, return this list
        if other is None or other.elt is None:
            return self
        # copy the list and get last element
        self = self.copy()
        last = self.get_last()
        # copy other list
        o = other.copy()
        # change next pointer of last element to other list and return 
        last.add_next(o)
        return self
    def add_next(self, val):
        """
        change the next pointer of list
        """
        self.next = val
    def print(self):
        """
        string representation of list
        """
        # while self is not none, print this value
        string = 'this list contains '
        while self != None:
            string += str(self.elt) + ' ' 
            # find next value
            self = self.next
        # print the string
        print(string)
    def car(self):
        """
        return first element aka this value
        """
        return self.elt
    def cdr(self):
        """
        return everything but first element, aka next
        """
        return self.next
    def length(self):
        """
        find length of list
        """
        # intiailze length value
        length = 0
        # while next exists, increment length and find next
        while self != None:
            length += 1
            self = self.next
        # return length
        return length
    def val_at_index(self, index):
        """
        find value at index in a list
        """
        # initialize place in list
        val = 0
        # while next exists and val is not index, find next
        while val != index and self != None:
            self = self.next
            # increment place in list aka val
            val += 1
        # return value at this index
        return self.elt
    def get_last(self):
        """
        get last element in the list
        """
        # while next exists, set self to next
        ls = self
        while ls.next != None:
            ls = ls.next
        # return last list
        return ls
    def copy(self):
        """
        copy the list
        """
        # copy first element
        new = LinkedList(self.elt)
        prev = new
        # while next exists, find next and copy element
        while self.next != None:
            self = self.next
            prev.next = LinkedList(self.elt)
            # change next pointer
            prev = prev.next
        # return copy list
        return new
    def map(self, function):
        """
        map a function to every element in list
        """
        # copy first element to this function mapped to list elt
        new = LinkedList(evaluate_function(function, [self.elt]))
        prev = new
        # while next exists, find next
        while self.next != None:
            self = self.next
            # set next pointer to a new list with this function applied to elt
            prev.next = LinkedList(evaluate_function(function, [self.elt]))
            prev = prev.next
        # return new list
        return new
    def filter(self, function):
        """
        filter elements of list by a function
        """
        # if the function applies, create a new list
        if evaluate_function(function, [self.elt]):
            new = LinkedList(self.elt)
        # else this is none
        else: new = None
        prev = new
        # while next exists, find next of list
        while self.next != None:
            self = self.next
            # create a new list with this function applied
            current = LinkedList(self.elt) if evaluate_function(function, [self.elt]) else None
            # if prev value was none, set prev to this current
            # if new is none, set new to prev
            if prev is None:
                prev = current
                if new is None: new = prev
            # if current value is none, don't change prev
            elif current is None:
                prev = prev
            # get last of prev and current to this list
            else:
                last = prev.get_last()
                last.add_next(current)
                prev = prev.next
        # return new
        return new
    def reduce(self, function, init):
        """
        reduce a list given a function and initial value
        """
        # get first val
        val = self.elt
        # apply function to first val and initial value
        result = evaluate_function(function, [init, val])
        # while next exists, find next
        while self.next != None:
            self = self.next
            val = self.elt
            # change result by applying function to this val and result
            result = evaluate_function(function, [result, val])
        # return result
        return result
        

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

def eq(args):
    # if all are equal in args, return true
    return all(x == args[0] for x in args)

def gtr(args):
    # make sure all elements are greater than next
    return all(last > n for last, n in zip(args, args[1:]))

def lst(args):
    # make sure all elements are less than next
    return all(last < n for last, n in zip(args, args[1:]))

def gtreq(args):
    # make sure any elements are not less than next
    return not any(last < n for last, n in zip(args, args[1:]))

def lsteq(args):
    # make sure any elements are not greater than next
    return not any(last > n for last, n in zip(args, args[1:]))

def AND(args, env):
    # return all(evaluate(a, env) for a in args)
    # if any of the args are not True, return False
    for a in args:
        if not evaluate(a, env):
            return False
    # return true if all pass
    return True

def OR(args, env):
    # return any(a for a in args)
    # if any of the args are True, return True
    for a in args:
        if evaluate(a, env):
            return True
    # return false if none pass
    return False

def NOT(args, env):
    # return not args[0]
    # return not of this value
    return not evaluate(args[0], env)

carlae_builtins = {
    '+': sum,
    '-': lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    '*': multiply,
    '/': divide,
    '=?': eq,
    '>': gtr,
    '>=': gtreq,
    '<': lst,
    '<=': lsteq,
    'and': AND,
    'or': OR,
    'not': NOT
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

def get_env(env, symbol):
    """
    helper to return env of symbol starting with env
    """
    # to not mutate env
    new_env = env
    # while env is not None
    while new_env != None:
        # if symbol exists in this env, return this env
        if symbol in new_env.env:
            return new_env
        # set env to parent
        else:
            new_env = new_env.parent
    # if not found raise Error
    raise EvaluationError

def evaluate_function(keyword, evl):
    """
    helper to evaluate a function given a keyword and the params
    to evaluate
    """
    # print(keyword)
    # print(keyword.params, evl)
    # if this is a basic keyword in builtins, evaluate builtin
    if keyword in carlae_builtins.values():
        # print('here')
        return keyword(evl)
    # if the parameter lengths do not match, raise Error
    elif len(keyword.params) != len(evl):
        raise EvaluationError('Parameters are not the same length')
    # if this is not a function, raise Error
    elif type(keyword) != Function:
        raise EvaluationError('This is not a function')
    else:
        # print(keyword, evl)
        # print('here')
        # create working environemnt
        this_env = Environment(keyword.env)
        # for every value to evaluate, define the parameter variable as this value
        for i in range(len(evl)):
            evaluate_helper(['define', keyword.params[i], evl[i]], this_env)
        # evaluate this function in the environment with updated parameters
        # print(keyword.function, keyword.params, this_env)
        return evaluate_helper(keyword.function, this_env)

def evaluate_helper(tree, env = None):
    """ 
    recursive helper for evaluate
    """
    # print(tree, type(tree))
    # if tree is simple int or float, return value
    if type(tree) == int or type(tree) == float or type(tree) == LinkedList:
        return tree
    # if tree is a string, try to return the value of the
    # var or raise an Exception if it does not exist
    elif type(tree) == str:
        if tree == '#t' or tree == '#f':
            return True if tree == '#t' else False
        if tree == '0' or tree == '0.0':
            return int(0) if tree == '0' else float(0.0)
        try:
            return get_env_part_2(env, tree)
        except:
            raise EvaluationError('This value does not exist')
    # if tree is a function return function
    elif type(tree) == Function:
        # print('hello i am also here')
        return tree
    # if tree is a list, do things with components
    elif type(tree) == list:
        # if tree is empty, raise error
        if not tree:
            raise EvaluationError('This is an empty list')
        # grab the keyword, var, and evl values
        try:
            keyword, var, evl = tree[0], tree[1], tree[2]
        # if there aren't three, just grab keyword and rest
        except:
            keyword, rest = tree[0], tree[1:]
        # print(keyword, var, evl)
        # if this is define
        # print(keyword)
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
        # conditionals
        elif keyword == 'if':
            # print('hello i am here')
            # evaluate the conditional 
            conditional = evaluate_helper(tree[1], env)
            true_exp, false_exp = tree[2], tree[3]
            # if conditional is true, evaluate true exp
            if conditional:
                return evaluate_helper(true_exp, env)
            # else evaluate false exp
            else:
                return evaluate_helper(false_exp, env)
        # and, or, not
        elif keyword == 'and' or keyword == 'or' or keyword == 'not':
            # find all of the conditionals in the list
            all_conditionals = tree[1:]
            # evaluate these conditionals
            function = evaluate_helper(keyword, env)
            # apply function (and, not, or) to conditionals
            return function(all_conditionals, env)
        # list functions
        elif keyword == 'list':
            # get args for list
            args = tree[1:]
            if args == []:
                # if there are no args, return a list of None
                return LinkedList(None)
            else:
                # make list of first element
                ls = LinkedList(evaluate_helper(args[0], env))
                prev = ls
                # for every other value in args
                for val in args[1:]:
                    # evaluate the value, create new linkedlist object
                    val = evaluate_helper(val, env)
                    new = LinkedList(val)
                    # set this new linkedlist as next of prev, change prev
                    prev.add_next(new)
                    prev = new
                # return list
                return ls
        elif keyword == 'car' or keyword == 'cdr':
            # evaluate the list in question
            ls = evaluate_helper(tree[1], env)
            # if it's not a list, raise error
            if type(ls) != LinkedList:
                raise EvaluationError('This is not a list')
            # if its empty raise error
            if ls.elt is None:
                raise EvaluationError('This is an empty list')
            # if its car, apply car
            if keyword == 'car':
                return ls.car()
            # if its cdr, apply cdr
            else:
                return ls.cdr()
        elif keyword == 'length':
            # evaluate the function at hand
            ls = evaluate_helper(tree[1], env)
            # if its not a list, raise error
            if type(ls) != LinkedList:
                raise EvaluationError('This is not a list')
            # if its empty, return 0
            if ls.elt is None:
                return 0
            # apply length to list
            return ls.length()
        elif keyword == 'elt-at-index':
            # evaluate list, get index
            ls = evaluate_helper(tree[1], env)
            index = tree[2]
            # if its not a list, raise error
            if type(ls) != LinkedList:
                raise EvaluationError('This is not a list')
            # if list is empty, raise error
            if ls.elt is None:
                raise EvaluationError('This is an empty list')
            # if index is greater than length, raise error
            if index >= ls.length():
                raise EvaluationError('Index out of bounds')
            # apply val_at_index to list
            return ls.val_at_index(index)
        elif keyword == 'concat':
            # if there are no args, return None
            if len(tree) == 1:
                return None
            # get first list
            ls = evaluate_helper(tree[1], env)
            start = ls
            # for rest of args
            for t in tree[2:]:
                # evaluate value
                new = evaluate_helper(t, env)
                # if start was None, set start to new
                if start is None:
                    start = new
                # concatenate lists (call __add__)
                else:
                    start = start + new
            # return start
            return start
        elif keyword == 'map':
            # evaluate function and list
            function = evaluate_helper(tree[1], env)
            ls = evaluate_helper(tree[2], env)
            # apply map to list
            new = ls.map(function)
            # return new copy
            return new
        elif keyword == 'filter':
            # evaluate function and list
            function = evaluate_helper(tree[1], env)
            ls = evaluate_helper(tree[2], env)
            # apply filter to list
            new = ls.filter(function)
            # return new copy
            return new
        elif keyword == 'reduce':
            # evaluate function, list, and init
            function = evaluate_helper(tree[1], env)
            ls = evaluate_helper(tree[2], env)
            init = evaluate_helper(tree[3], env)
            # apply reduce to list
            return ls.reduce(function, init)
        # for begin
        elif keyword == 'begin':
            # get args and last index
            args = tree[1:][:]
            last = len(args) - 1
            # iterate through args and indices
            for index, arg in enumerate(args):
                # if at end, return this evaluated
                if index == last:
                    return evaluate_helper(arg, env)
                # evaluate current args
                evaluate_helper(arg, env)
        # for let
        elif keyword == 'let':
            # get args, body, and create new env with this as parent
            args = tree[1]
            body = tree[2]
            this_env = Environment(env)
            # for all args
            for arg in args:
                # get var and val
                var, val = arg
                # evaluate value 
                val = evaluate_helper(val, env)
                # add this to new current env
                this_env.add_val(var, val)
            # return the body evalulated in this env
            return evaluate_helper(body, this_env)
        # for set!
        elif keyword == 'set!':
            # get var and expr
            var = tree[1]
            expr = evaluate_helper(tree[2], env)
            # find env of var
            new_env = get_env(env, var)
            # change env
            new_env.add_val(var, expr)
            return expr
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
                # print('here is function', function)
                # for every value in the rest which is a paren
                for val in tree[1:]:
                    # evaluate this val to get value
                    v = evaluate_helper(val, env)
                    # add this to the parameters
                    params.append(v)
                # evaluate this function with these parameters
                # print(function, params)
                # print('stuff is ', function, params)
                return evaluate_function(function, params)
            except:
                # raise Error if this doesn't work for some reason
                raise EvaluationError('Parameters are not the same length')
    else:
        # raise Error if nothing works
        raise EvaluationError('You fucked up')

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

def evaluate_file(filename, env=None):
    """
    function to evaluate files
    """
    # open the file and join all lines
    f = open(filename, 'r')
    exp = ' '.join(line.strip() for line in f)
    f.close()
    # evaluate and return
    result = evaluate(parse(tokenize(exp)), env)
    return result

x = "(begin (define (foo bar) (lambda (x y) (- bar x y))) (define bar 7) (define something (foo 6)) (list (something 2 3) ((foo 9) 8 7)))"

def REPL(env = None):
    """
    Read, Evaluate, Print Loop for testing code
    """
    # grab input
    inp = input('inpt > ')
    # while not quit
    if env is None:
        env = create_new_env(None)
    while inp != 'QUIT' and inp != 'quit':
        # try to evaluate input
        try:
            print('  outpt > ', evaluate(parse(tokenize(inp)), env))
        # catch exception, print
        except:
            e = sys.exc_info()[0]
            print('  outpt: ', e)
        # grab next output
        # print('  outpt > ', evaluate(parse(tokenize(inp)), env))
        inp = input('inpt > ')

if __name__ == '__main__':
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)
    # pass
    env = create_new_env(None)
    args = sys.argv
    for arg in args[1:]:
        # evaluate(parse(tokenize(arg)), env)
        evaluate_file(arg, env)
    REPL(env)
