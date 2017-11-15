#!/usr/bin/env python3
import os
import lab
import sys
import json
import unittest

TEST_DIRECTORY = os.path.dirname(__file__)


def list_from_ll(ll):
    if isinstance(ll, lab.LinkedList):
        if ll.next is None:
            return [list_from_ll(ll.elt)]
        return [list_from_ll(ll.elt)] + list_from_ll(ll.next)
    elif isinstance(ll, (float, int)):
        return ll
    else:
        return 'SOMETHING'

class LispTest(unittest.TestCase):
    @staticmethod
    def make_tester(func):
        """
        Helper to wrap a function so that, when called, it produces a
        dictionary instead of its normal result.  If the function call works
        without raising an exception, then the results are included.
        Otherwise, the dictionary includes information about the exception that
        was raised.
        """
        def _tester(*args):
            try:
                return {'ok': True, 'output': func(*args)}
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                return {'ok': False, 'type': exc_type.__name__}
        return _tester

    @staticmethod
    def load_test_values(n):
        """
        Helper function to load test inputs/outputs
        """
        with open('test_inputs/%s.json' % n) as f:
            inputs = json.load(f)
        with open('test_outputs/%s.json' % n) as f:
            outputs = json.load(f)
        return inputs, outputs

    @staticmethod
    def _test_file(fname, num):
        try:
            out = lab.evaluate_file(os.path.join('test_files', fname))
            out = list_from_ll(out)
            out = {'ok': True, 'output': out}
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            out = {'ok': False, 'type': exc_type.__name__}
        with open('test_outputs/%s.json' % num) as f:
            expected = json.load(f)
        return out, expected

    @staticmethod
    def run_continued_evaluations(ins):
        """
        Helper to evaluate a sequence of expressions in an environment.
        """
        env = None
        outs = []
        try:
            t = LispTest.make_tester(lab.result_and_env)
        except:
            t = LispTest.make_tester(lab.evaluate)
        for i in ins:
            if env is None:
                args = (i, )
            else:
                args = (i, env)
            out = t(*args)
            if out['ok']:
                env = out['output'][1]
            if out['ok']:
                try:
                    typecheck = (int, float, lab.LinkedList)
                    func = list_from_ll
                except:
                    typecheck = (int, float)
                    func = lambda x: x if isinstance(x, typecheck) else 'SOMETHING'
                out['output'] = func(out['output'][0])
            outs.append(out)
        return outs

    def _compare_outputs(self, x, y):
        self.assertEqual(x['ok'], y['ok'])
        if x['ok']:
            if isinstance(x['output'], (int, float)):
                self.assertAlmostEqual(x['output'], y['output'])
            else:
                self.assertEqual(x['output'], y['output'])
        else:
            self.assertEqual(x, y)

    def _test_continued_evaluations(self, n):
        """
        Test that the results from running continued evaluations in the same
        environment match the expected values.
        """
        inp, out = self.load_test_values(n)
        results = self.run_continued_evaluations(inp)
        for result, expected in zip(results, out):
            self._compare_outputs(result, expected)

    def run_test_number(self, n, func):
        tester = self.make_tester(func)
        inp, out = self.load_test_values(n)
        for i, o in zip(inp, out):
            self._compare_outputs(tester(i), o)


class Test1_OldTests(LispTest):
    def test_01_tokenize(self):
        self.run_test_number(1, lab.tokenize)

    def test_02_parse(self):
        self.run_test_number(2, lab.parse)

    def test_03_tokenize_and_parse(self):
        self.run_test_number(3, lambda i: lab.parse(lab.tokenize(i)))

    def test_04_calc(self):
        self.run_test_number(4, lab.evaluate)

    def test_05_mult_div(self):
        self.run_test_number(5, lab.evaluate)

    def test_06_simple_assignment(self):
        self._test_continued_evaluations(6)

    def test_07_simple_assignment(self):
        self._test_continued_evaluations(7)

    def test_08_bad_lookups(self):
        self._test_continued_evaluations(8)

    def test_09_rename_builtin(self):
        self._test_continued_evaluations(9)

    def test_10_simple_function(self):
        self._test_continued_evaluations(10)

    def test_11_inline_lambda(self):
        self._test_continued_evaluations(11)

    def test_12_closures(self):
        self._test_continued_evaluations(12)

    def test_13_short_definition(self):
        self._test_continued_evaluations(13)

    def test_14_dependent_definition(self):
        self._test_continued_evaluations(14)

    def test_15_scoping_1(self):
        self._test_continued_evaluations(15)

    def test_16_scoping_2(self):
        self._test_continued_evaluations(16)

    def test_17_scoping_3(self):
        self._test_continued_evaluations(17)

    def test_18_scoping_4(self):
        self._test_continued_evaluations(18)

    def test_19_scoping_5(self):
        self._test_continued_evaluations(19)

    def test_20_calling_errors(self):
        self._test_continued_evaluations(20)

    def test_21_functionception(self):
        self._test_continued_evaluations(21)

    def test_22_alias(self):
        self._test_continued_evaluations(22)

    def test_23_big_scoping_1(self):
        self._test_continued_evaluations(23)

    def test_24_big_scoping_2(self):
        self._test_continued_evaluations(24)

    def test_25_big_scoping_3(self):
        self._test_continued_evaluations(25)

    def test_26_big_scoping_4(self):
        self._test_continued_evaluations(26)


class Test2_NewTestsForOldBehaviors(LispTest):
    def test_27_goofy_names(self):
        self._test_continued_evaluations(27)

    def test_28_nested_defines(self):
        self._test_continued_evaluations(28)

    def test_29_shadow_builtin(self):
        self._test_continued_evaluations(29)


class Test3_Conditionals(LispTest):
    def test_30_conditionals(self):
        self._test_continued_evaluations(30)

    def test_31_abs(self):
        self._test_continued_evaluations(31)

    def test_32_and(self):
        self._test_continued_evaluations(32)

    def test_33_or(self):
        self._test_continued_evaluations(33)

    def test_34_not(self):
        self._test_continued_evaluations(34)

    def test_35_shortcircuit_1(self):
        self._test_continued_evaluations(35)

    def test_36_shortcircuit_2(self):
        self._test_continued_evaluations(36)

    def test_37_shortcircuit_3(self):
        self._test_continued_evaluations(37)

    def test_38_shortcircuit_4(self):
        self._test_continued_evaluations(38)

    def test_39_conditional_scoping(self):
        self._test_continued_evaluations(39)

    def test_40_conditional_scoping_2(self):
        self._test_continued_evaluations(40)


class Test4_Lists(LispTest):
    def test_41_lists(self):
        self._test_continued_evaluations(41)

    def test_42_car_cdr(self):
        self._test_continued_evaluations(42)

    def test_43_car_cdr_2(self):
        self._test_continued_evaluations(43)

    def test_44_length(self):
        self._test_continued_evaluations(44)

    def test_45_indexing(self):
        self._test_continued_evaluations(45)

    def test_46_concat(self):
        self._test_continued_evaluations(46)

    def test_47_list_ops(self):
        self._test_continued_evaluations(47)

    def test_48_map_builtin(self):
        self._test_continued_evaluations(48)

    def test_49_map_carlaefunc(self):
        self._test_continued_evaluations(49)

    def test_50_filter_builtin(self):
        self._test_continued_evaluations(50)

    def test_51_filter_carlaefunc(self):
        self._test_continued_evaluations(51)

    def test_52_reduce_builtin(self):
        self._test_continued_evaluations(52)

    def test_53_reduce_carlaefunc(self):
        self._test_continued_evaluations(53)

    def test_54_map_filter_reduce(self):
        self._test_continued_evaluations(54)


class Test5_Let_SetBang_Begin(LispTest):
    def test_55_let(self):
        self._test_continued_evaluations(55)

    def test_56_let_2(self):
        self._test_continued_evaluations(56)

    def test_57_let_3(self):
        self._test_continued_evaluations(57)

    def test_58_setbang(self):
        self._test_continued_evaluations(58)

    def test_59_begin(self):
        self._test_continued_evaluations(59)

    def test_60_begin2(self):
        self._test_continued_evaluations(60)


class Test6_Files(LispTest):
    def test_61(self):
        self._compare_outputs(*self._test_file("simple_test1.crl", 61))

    def test_62(self):
        self._compare_outputs(*self._test_file("simple_test2.crl", 62))

    def test_63(self):
        self._compare_outputs(*self._test_file("simple_test3.crl", 63))

    def test_64(self):
        self._compare_outputs(*self._test_file("simple_test4.crl", 64))

    def test_65(self):
        self._compare_outputs(*self._test_file("simple_test5.crl", 65))


class Test7_DeepNesting(LispTest):
    def test_66(self):
        self._test_continued_evaluations(66)

    def test_67(self):
        self._test_continued_evaluations(67)

    def test_68(self):
        self._test_continued_evaluations(68)


class Test8_RealPrograms(LispTest):
    def test_69_counters_oop(self):
        self._test_continued_evaluations(69)

    def test_70_fizzbuzz(self):
        self._test_continued_evaluations(70)

    def test_71_primes(self):
        self._test_continued_evaluations(71)

    def test_72_averages_oop(self):
        self._test_continued_evaluations(72)


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
