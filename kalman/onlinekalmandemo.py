from onlinekalman import MultiOnlineKalman

k = MultiOnlineKalman()
print(k.take_multiple_observations([
    [100, 100, 100],
    [200, 200, 200],
    [300, 300, 300]
]))

print(k.take_multiple_observations([
    [100, 100, 101],
    [200, 200, 201],
    [300, 300, 301]
]))

print(k.take_multiple_observations([
    [100, 100, 102],
    [200, 200, 202],
    [300, 300, 302]
]))

print(k.take_multiple_observations([
    [100, 100, 103],
    [300, 300, 303],
    [200, 200, 203]
]))