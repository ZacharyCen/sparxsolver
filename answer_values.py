"""Exact, bounded numeric/surd comparison without eval or executable expressions."""
import ast
from fractions import Fraction
import re


def _root(value):
    if value < 0 or value.numerator * value.denominator > 10**8:
        raise ValueError('Unsupported radicand')
    number = value.numerator * value.denominator
    outside, inside, factor = 1, number, 2
    while factor*factor <= inside:
        while inside and inside % (factor*factor) == 0:
            outside *= factor
            inside //= factor*factor
        factor += 1
    return inside, Fraction(outside, value.denominator)


def _add(left, right, sign=1):
    result = dict(left)
    for root, coefficient in right.items():
        result[root] = result.get(root, Fraction(0)) + sign*coefficient
    return {root: coefficient for root, coefficient in result.items() if coefficient}


def _multiply(left, right):
    result = {}
    for a, x in left.items():
        for b, y in right.items():
            root, coefficient = _root(Fraction(a*b))
            result[root] = result.get(root, Fraction(0)) + x*y*coefficient
    return {root: coefficient for root, coefficient in result.items() if coefficient}


def _tex(source):
    """Expand braced TeX fractions before parsing a restricted arithmetic tree."""
    def group(index):
        while index < len(source) and source[index].isspace(): index += 1
        if index >= len(source) or source[index] != '{': raise ValueError('Missing argument')
        start, depth = index+1, 1
        index += 1
        while index < len(source) and depth:
            depth += (source[index]=='{') - (source[index]=='}')
            index += 1
        if depth: raise ValueError('Unbalanced argument')
        return _tex(source[start:index-1]), index
    result, index = [], 0
    while index < len(source):
        match = re.match(r'\\(?:dfrac|tfrac|frac)', source[index:])
        if match:
            first,index = group(index+len(match[0]))
            second,index = group(index)
            result.append('(('+first+')/('+second+'))')
        else:
            result.append(source[index]);index += 1
    return ''.join(result)


def expression_value(text):
    """Canonical rational/square-root value, or None for unreadable/unsupported input."""
    try:
        source = str(text).strip().replace('−','-').replace('–','-').replace('×','*').replace('÷','/')
        source = re.sub(r'^[A-Za-z]\s*=\s*', '', source)
        if len(source)>256: return None
        source = _tex(source).replace(r'\left','').replace(r'\right','')
        source = source.replace(r'\sqrt','sqrt').replace('{','(').replace('}',')')
        source = source.replace(r'\times','*').replace(r'\cdot','*').replace(r'\div','/')
        source = re.sub(r'√\s*(\d+(?:\.\d+)?)', r'sqrt(\1)', source)
        source = source.replace('√(', 'sqrt(').replace('^','**')
        source = re.sub(r'(\d|\))\s*(?=sqrt\b|\()', r'\1*', source)
        source = re.sub(r'\s+', '', source)
        if not re.fullmatch(r'[0-9.\s()+*/\-sqrt]+', source): return None
        tree = ast.parse(source, mode='eval')
        if len(list(ast.walk(tree)))>80: return None
        def visit(node):
            if isinstance(node,ast.Expression): return visit(node.body)
            if isinstance(node,ast.Constant) and type(node.value) in (int,float):
                literal=ast.get_source_segment(source,node)
                if not literal or len(literal)>64 or not re.fullmatch(r'\d+(?:\.\d*)?|\.\d+',literal): raise ValueError('Invalid number')
                value=Fraction(literal)
                if abs(value)>10**64: raise ValueError('Number too large')
                return {1:value} if value else {}
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
                value=visit(node.operand)
                return {r:(-c if isinstance(node.op,ast.USub) else c) for r,c in value.items()}
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='sqrt' and len(node.args)==1 and not node.keywords:
                value=visit(node.args[0])
                if set(value)-{1}: raise ValueError('Nested irrational root')
                root,coefficient=_root(value.get(1,Fraction(0)))
                return {root:coefficient} if root else {}
            if isinstance(node,ast.BinOp):
                left,right=visit(node.left),visit(node.right)
                if isinstance(node.op,ast.Add): return _add(left,right)
                if isinstance(node.op,ast.Sub): return _add(left,right,-1)
                if isinstance(node.op,ast.Mult): return _multiply(left,right)
                if isinstance(node.op,ast.Div) and set(right)<={1} and right.get(1):
                    return {r:c/right[1] for r,c in left.items()}
                if isinstance(node.op,ast.Pow) and set(right)<={1}:
                    power=right.get(1,Fraction(0))
                    if power.denominator!=1 or not 0<=power<=8: raise ValueError('Unsupported power')
                    result={1:Fraction(1)}
                    for _ in range(int(power)): result=_multiply(result,left)
                    return result
            raise ValueError('Unsupported expression')
        result=visit(tree)
        if len(result)>16 or any(c.numerator.bit_length()>512 or c.denominator.bit_length()>512 for c in result.values()): return None
        return tuple(sorted(result.items()))
    except (ValueError,TypeError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):
        return None


def entry_matches(actual: str, expected: str) -> bool:
    """Reject malformed decimal entry even if its arithmetic value is equivalent."""
    source=str(actual).strip().replace('−','-').replace('–','-')
    if re.fullmatch(r'[+-]?0\d+(?:\.\d*)?',source):return False
    value=expression_value(source)
    return value is not None and value==expression_value(expected)
