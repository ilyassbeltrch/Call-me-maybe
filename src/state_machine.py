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


class JSONStateMachine:
    def __init__(
        self,
        id_to_str: dict[int, str],
        function_names: list[str],
        param_names: list[str],
        param_types: dict[str, str],
    ) -> None:
        self.keys_used: set[str] = set()
        self.id_to_str = id_to_str
        self.str_to_id: dict[str, int] = {v: k for k, v in id_to_str.items()}
        self.function_names = function_names
        self.param_names = param_names
        self.param_types = param_types

        self.state = State.START
        self.built_so_far = ""
        self.current_key = ""
        self.current_param_idx = 0
        self.current_param_built = ""
        self.current_val_built = ""

    def _force(self, s: str) -> list[int]:
        tid = self.str_to_id.get(s)
        return [tid] if tid is not None else []

    def _tokens_continuing(self, target: str, built: str) -> list[int]:
        if not target.startswith(built):
            return []
        remaining = target[len(built):]
        valid = []
        for token_id, token_str in self.id_to_str.items():
            if token_str and remaining.startswith(token_str):
                valid.append(token_id)
        return valid

    def _tokens_continuing_any(self, targets: list[str], built: str) -> list[int]:
        valid_set: set[int] = set()
        for target in targets:
            valid_set.update(self._tokens_continuing(target, built))
        return list(valid_set)

    def _number_tokens(self) -> list[int]:
        valid = []
        has_dot = "." in self.current_val_built
        for token_id, token_str in self.id_to_str.items():
            if not token_str:
                continue
            if all(c in "0123456789." for c in token_str):
                if "." in token_str and has_dot:
                    continue
                valid.append(token_id)
        return valid

    def _string_val_tokens(self) -> list[int]:
        valid = []
        for token_id, token_str in self.id_to_str.items():
            if token_str and '"' not in token_str:
                valid.append(token_id)
        return valid

    def _is_number_complete(self) -> bool:
        try:
            float(self.current_val_built)
            return True
        except ValueError:
            return False

    def _has_more_params(self) -> bool:
        return self.current_param_idx < len(self.param_names) - 1

    def get_valid_ids(self) -> list[int]:
        s = self.state

        if s == State.START:
            return self._force("{")
        if s == State.KEY_OPEN:
            return self._force('"')
        if s == State.KEY_NAME:
            if "name" not in self.keys_used:
                return self._tokens_continuing("name", self.built_so_far)
            return self._tokens_continuing("parameters", self.built_so_far)
        if s == State.KEY_CLOSE:
            return self._force('"')
        if s == State.COLON:
            return self._force(":")
        if s == State.VAL_STR_OPEN:
            return self._force('"')
        if s == State.VAL_STR_BODY:
            return self._tokens_continuing_any(self.function_names, self.built_so_far)
        if s == State.VAL_STR_CLOSE:
            return self._force('"')
        if s == State.OUTER_NEXT:
            return self._force(",")
        if s == State.VAL_OBJ_OPEN:
            return self._force("{")
        if s == State.PARAM_KEY_OPEN:
            return self._force('"')
        if s == State.PARAM_KEY_BODY:
            current_param = self.param_names[self.current_param_idx]
            return self._tokens_continuing(current_param, self.current_param_built)
        if s == State.PARAM_KEY_CLOSE:
            return self._force('"')
        if s == State.PARAM_COLON:
            return self._force(":")
        if s == State.PARAM_VAL_NUM:
            valid = self._number_tokens()
            if self._is_number_complete() and len(self.current_val_built) > 0:
                valid += self._force(",") if self._has_more_params() else self._force("}")
            return valid
        if s == State.PARAM_VAL_STR_OPEN:
            return self._force('"')
        if s == State.PARAM_VAL_STR_BODY:
            return self._string_val_tokens() + self._force('"')
        if s == State.PARAM_NEXT:
            return self._force(",") if self._has_more_params() else self._force("}")
        if s == State.FINAL_CLOSE:
            return self._force("}")
        return []

    def update(self, token_id: int) -> None:
        token_str = self.id_to_str.get(token_id, "")
        self._transition(token_str)

    def _transition(self, token_str: str) -> None:
        s = self.state

        if s == State.START:
            self.state = State.KEY_OPEN

        elif s == State.KEY_OPEN:
            self.state = State.KEY_NAME
            self.built_so_far = ""

        elif s == State.KEY_NAME:
            self.built_so_far += token_str
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
            if self.current_key == "name":
                self.state = State.VAL_STR_OPEN
            else:
                self.state = State.VAL_OBJ_OPEN

        elif s == State.VAL_STR_OPEN:
            self.state = State.VAL_STR_BODY
            self.built_so_far = ""

        elif s == State.VAL_STR_BODY:
            self.built_so_far += token_str
            if self.built_so_far in self.function_names:
                self.state = State.VAL_STR_CLOSE

        elif s == State.VAL_STR_CLOSE:
            self.state = State.OUTER_NEXT

        elif s == State.OUTER_NEXT:
            self.state = State.KEY_OPEN

        elif s == State.VAL_OBJ_OPEN:
            if self.param_names:
                self.state = State.PARAM_KEY_OPEN
            else:
                self.state = State.PARAM_NEXT

        elif s == State.PARAM_KEY_OPEN:
            self.state = State.PARAM_KEY_BODY
            self.current_param_built = ""

        elif s == State.PARAM_KEY_BODY:
            self.current_param_built += token_str
            current_param = self.param_names[self.current_param_idx]
            if self.current_param_built == current_param:
                self.state = State.PARAM_KEY_CLOSE

        elif s == State.PARAM_KEY_CLOSE:
            self.state = State.PARAM_COLON

        elif s == State.PARAM_COLON:
            current_param = self.param_names[self.current_param_idx]
            ptype = self.param_types.get(current_param, "string")
            self.current_val_built = ""
            if ptype == "number":
                self.state = State.PARAM_VAL_NUM
            else:
                self.state = State.PARAM_VAL_STR_OPEN

        elif s == State.PARAM_VAL_NUM:
            if token_str == ",":
                self.current_param_idx += 1
                self.state = State.PARAM_KEY_OPEN
            elif token_str == "}":
                self.state = State.FINAL_CLOSE
            else:
                self.current_val_built += token_str

        elif s == State.PARAM_VAL_STR_OPEN:
            self.state = State.PARAM_VAL_STR_BODY
            self.current_val_built = ""

        elif s == State.PARAM_VAL_STR_BODY:
            if token_str == '"':
                self.state = State.PARAM_NEXT
            else:
                self.current_val_built += token_str

        elif s == State.PARAM_NEXT:
            if token_str == ",":
                self.current_param_idx += 1
                self.state = State.PARAM_KEY_OPEN
            elif token_str == "}":
                self.state = State.FINAL_CLOSE

        elif s == State.FINAL_CLOSE:
            self.state = State.DONE

        elif s == State.DONE:
            pass