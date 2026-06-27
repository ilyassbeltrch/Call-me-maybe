from enum import Enum, auto


class State(Enum):
    START = auto()
    KEY_OPEN = auto()
    KEY_NAME = auto()
    KEY_CLOSE = auto()
    COLON = auto()
    VAL_STR_OPEN = auto()
    VAL_STR_BODY = auto()
    VAL_STR_CLOSE = auto()
    OUTER_NEXT = auto()
    VAL_OBJ_OPEN = auto()
    PARAM_KEY_OPEN = auto()
    PARAM_KEY_BODY = auto()
    PARAM_KEY_CLOSE = auto()
    PARAM_COLON = auto()
    PARAM_VAL_NUM = auto()
    PARAM_VAL_STR_OPEN = auto()
    PARAM_VAL_STR_BODY = auto()
    PARAM_NEXT = auto()
    FINAL_CLOSE = auto()
    DONE = auto()


MAX_STRING_LEN = 200


class JSONStateMachine:
    def __init__(
        self,
        id_to_str: dict[int, str],
        function_name: str,
        param_names: list[str],
        param_types: dict[str, str],
    ) -> None:
        self.id_to_str = id_to_str
        self.str_to_id: dict[str, int] = {v: k for k, v in id_to_str.items()}
        self.function_name = function_name
        self.param_names = param_names
        self.param_types = param_types

        self.state = State.START
        self.keys_used: set[str] = set()
        self.built_so_far = ""
        self.current_key = ""
        self.current_param_idx = 0
        self.current_param_built = ""
        self.current_val_built = ""

        self._number_tokens_no_dot: list[int] = []
        self._number_tokens_with_dot: list[int] = []
        self._dot_token: list[int] = []
        self._string_tokens: list[int] = []

        for tid, s in id_to_str.items():
            if not s:
                continue
            # number tokens: digits and dot only
            if all(c in "0123456789." for c in s):
                self._number_tokens_no_dot.append(tid)
                if "." not in s:
                    self._number_tokens_with_dot.append(tid)

            # string tokens: block control chars, backslash, and quote only
            # allow { } so paths/templates work
            has_control = any(ord(c) < 0x20 for c in s)
            has_quote = '"' in s
            has_backslash = '\\' in s
            if not has_control and not has_quote and not has_backslash:
                self._string_tokens.append(tid)

    def _force(self, s: str) -> list[int]:
        tid = self.str_to_id.get(s)
        return [tid] if tid is not None else []

    def _continuing(self, target: str, built: str) -> list[int]:
        if not target.startswith(built):
            return []
        remaining = target[len(built):]
        return [
            tid for tid, s in self.id_to_str.items()
            if s and remaining.startswith(s)
        ]

    def _has_more_params(self) -> bool:
        return self.current_param_idx < len(self.param_names) - 1

    def _number_complete(self) -> bool:
        try:
            float(self.current_val_built)
            return True
        except ValueError:
            return False

    def get_valid_ids(self) -> list[int]:
        s = self.state

        if s == State.START:
            return self._force("{")
        if s in (
            State.KEY_OPEN,
            State.KEY_CLOSE,
            State.VAL_STR_OPEN,
            State.VAL_STR_CLOSE,
            State.PARAM_KEY_OPEN,
            State.PARAM_KEY_CLOSE,
            State.PARAM_VAL_STR_OPEN
        ):
            return self._force('"')
        if s == State.KEY_NAME:
            target = "name" if "name" not in self.keys_used else "parameters"
            return self._continuing(target, self.built_so_far)
        if s == State.COLON:
            return self._force(":")
        if s == State.VAL_STR_BODY:
            return self._continuing(self.function_name, self.built_so_far)
        if s == State.OUTER_NEXT:
            return self._force(",")
        if s == State.VAL_OBJ_OPEN:
            return self._force("{")
        if s == State.PARAM_KEY_BODY:
            target = self.param_names[self.current_param_idx]
            return self._continuing(target, self.current_param_built)
        if s == State.PARAM_COLON:
            return self._force(":")
        if s == State.PARAM_VAL_NUM:
            has_dot = "." in self.current_val_built
            valid = (
                self._number_tokens_with_dot
                if has_dot
                else self._number_tokens_no_dot
            )
            if self._number_complete() and self.current_val_built:
                # Force a dot if no dot yet to ensure float output
                if "." not in self.current_val_built:
                    valid = valid + self._force(".")
                valid = valid + (
                    self._force(",")
                    if self._has_more_params()
                    else self._force("}")
                )
            return valid
        if s == State.PARAM_VAL_STR_BODY:
            if len(self.current_val_built) >= MAX_STRING_LEN:
                return self._force('"')
            return self._string_tokens + self._force('"')
        if s == State.PARAM_NEXT:
            return (
                self._force(",")
                if self._has_more_params()
                else self._force("}")
            )
        if s == State.FINAL_CLOSE:
            return self._force("}")
        return []

    def update(self, token_id: int) -> None:
        t = self.id_to_str.get(token_id, "")
        s = self.state

        if s == State.START:
            self.state = State.KEY_OPEN

        elif s == State.KEY_OPEN:
            self.state = State.KEY_NAME
            self.built_so_far = ""

        elif s == State.KEY_NAME:
            self.built_so_far += t
            if self.built_so_far == "name":
                self.current_key = "name"
                self.keys_used.add("name")
                self.state = State.KEY_CLOSE
            elif self.built_so_far == "parameters":
                self.current_key = "parameters"
                self.keys_used.add("parameters")
                self.state = State.KEY_CLOSE

        elif s == State.KEY_CLOSE:
            self.state = State.COLON

        elif s == State.COLON:
            self.state = (
                State.VAL_STR_OPEN
                if self.current_key == "name"
                else State.VAL_OBJ_OPEN
            )

        elif s == State.VAL_STR_OPEN:
            self.state = State.VAL_STR_BODY
            self.built_so_far = ""

        elif s == State.VAL_STR_BODY:
            self.built_so_far += t
            if self.built_so_far == self.function_name:
                self.state = State.VAL_STR_CLOSE

        elif s == State.VAL_STR_CLOSE:
            self.state = State.OUTER_NEXT

        elif s == State.OUTER_NEXT:
            self.state = State.KEY_OPEN

        elif s == State.VAL_OBJ_OPEN:
            self.state = (
                State.PARAM_KEY_OPEN
                if self.param_names
                else State.PARAM_NEXT
            )

        elif s == State.PARAM_KEY_OPEN:
            self.state = State.PARAM_KEY_BODY
            self.current_param_built = ""

        elif s == State.PARAM_KEY_BODY:
            self.current_param_built += t
            target = self.param_names[self.current_param_idx]
            if self.current_param_built == target:
                self.state = State.PARAM_KEY_CLOSE

        elif s == State.PARAM_KEY_CLOSE:
            self.state = State.PARAM_COLON

        elif s == State.PARAM_COLON:
            target = self.param_names[self.current_param_idx]
            ptype = self.param_types.get(target, "string")
            self.current_val_built = ""
            self.state = (
                State.PARAM_VAL_NUM
                if ptype == "number"
                else State.PARAM_VAL_STR_OPEN
            )

        elif s == State.PARAM_VAL_NUM:
            if t == ",":
                self.current_param_idx += 1
                self.state = State.PARAM_KEY_OPEN
            elif t == "}":
                self.state = State.FINAL_CLOSE
            else:
                self.current_val_built += t

        elif s == State.PARAM_VAL_STR_OPEN:
            self.state = State.PARAM_VAL_STR_BODY
            self.current_val_built = ""

        elif s == State.PARAM_VAL_STR_BODY:
            if t == '"':
                self.state = State.PARAM_NEXT
            else:
                self.current_val_built += t

        elif s == State.PARAM_NEXT:
            if t == ",":
                self.current_param_idx += 1
                self.state = State.PARAM_KEY_OPEN
            elif t == "}":
                self.state = State.FINAL_CLOSE

        elif s == State.FINAL_CLOSE:
            self.state = State.DONE