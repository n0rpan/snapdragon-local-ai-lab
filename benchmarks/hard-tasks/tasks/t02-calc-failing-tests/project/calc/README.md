# calc

`evaluate(text)` evaluates an arithmetic expression and returns an `int` or a `float`.
Invalid input raises `CalcError(message, pos)`, where `pos` is the 0-based index of the
offending character or token (for unexpected end of input: `len(text)`).

## Tokens

- Whitespace is ignored.
- Numbers: `12`, `3.25`, `.5`, `1e3`, `2.5E-2`, `5e+2`. An exponent is `e` or `E`, an optional
  `+` or `-`, then one or more digits. An `e` not followed by that is an unexpected character.
- Operators `+ - * / ^` and parentheses `( )`.
- Any other character is an unexpected character.

## Grammar (lowest to highest precedence)

    expr  := term (('+' | '-') term)*
    term  := unary (('*' | '/') unary)*
    unary := ('+' | '-') unary | power
    power := atom ('^' unary)?
    atom  := NUMBER | '(' expr ')'

So `^` is right-associative (`2^3^2` is `512`), binds tighter than unary minus (`-2^2` is `-4`),
and its exponent may carry unary signs (`2^-1` is `0.5`). `--2` is `2`, `2*-3` is `-6`.

## Result types

- A number literal is `int` if it contains only digits (`12`), otherwise `float` (`.5`, `3.25`, `1e3`).
- `+`, `-`, `*` give `int` when both operands are `int`, otherwise `float`.
- `/` is always true division and gives `float`.
- `^` gives `int` when both operands are `int` and the exponent is not negative, otherwise `float`.
- Unary `+` and `-` keep the type of their operand. Parentheses do not change the type.

## Errors

- Unexpected character (e.g. `$`): its index.
- Unexpected token (e.g. `1)`, `2 2`, `*2`): index of that token.
- Unexpected end of input (e.g. `2+`, `(1`, empty text): `len(text)`.
- Division by zero (e.g. `2 / 0`): index of the `/` operator.
