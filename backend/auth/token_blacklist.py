class TokenBlacklist:
    def add(self, *args, **kwargs): pass
    def __contains__(self, *args, **kwargs): return False
token_blacklist = TokenBlacklist()
