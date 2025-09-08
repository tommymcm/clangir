// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | cir-opt --cir-points-to-diagnostics --verify-diagnostics -o /dev/null

struct Node {
  // expected-remark@above {{load { this }}}
  // expected-remark@above {{store { this }}}
  // expected-remark@above {{load {  }}}
  // expected-remark@above {{store {  }}}
  // expected-remark@above {{load { :unknown: }}}
  // expected-remark@above {{store { :unknown: }}}
  // expected-remark@above {{copy from { :unknown: } to { :unknown: }}}
  // expected-remark@above {{move from { :unknown: } to { :unknown: }}}
  // expected-remark@above {{load { __retval }}}
  // expected-remark@above {{store { __retval }}}

  int val;
  // expected-remark@above {{store { :unknown: }}}
  // expected-remark@above {{read from { copy }}}
  // expected-remark@above {{read from { move }}}
  // expected-remark@above {{read from { :unknown: }}}
};

int test_copy_ctor() {
  Node orig;
  Node copy(orig);
  // expected-remark@above {{copy from { orig } to { copy }}}

  return copy.val;
  // expected-remark@above {{load { copy }}}
  // expected-remark@above {{load { __retval }}}
  // expected-remark@above {{store { __retval }}}
}

int test_move_ctor() {
  Node orig;
  Node move((Node &&)orig);
  // expected-remark@above {{move from { orig } to { move }}}

  return move.val;
  // expected-remark@above {{load { move }}}
  // expected-remark@above {{load { __retval }}}
  // expected-remark@above {{store { __retval }}}
}

int test_copy_assign() {
  Node orig, copy;
  copy = orig;
  // expected-remark@above {{copy from { orig } to { copy }}}

  return copy.val;
  // expected-remark@above {{load { copy }}}
  // expected-remark@above {{load { __retval }}}
  // expected-remark@above {{store { __retval }}}
}

int test_move_assign() {
  Node orig, move;
  move = (Node &&)orig;
  // expected-remark@above {{move from { orig } to { move }}}

  return move.val;
  // expected-remark@above {{load { move }}}
  // expected-remark@above {{load { __retval }}}
  // expected-remark@above {{store { __retval }}}
}
