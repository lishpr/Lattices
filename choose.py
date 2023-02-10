import random

def select_lattice(n, m):
    if n % 2 == 0 or n <= 4:
        return None
    if m % 8 != 0:
        return None
    
    lattice = []
    for x in range(n//2):
        for y in range(n//2):
            for z in range(n//2):
                lattice.append((x, y, z))
    
    random.shuffle(lattice)
    lattice = lattice[:m//8]
    
    result = []
    for x, y, z in lattice:
        result.append(coordinates_to_number((x, y, z), n))
        result.append(coordinates_to_number((x, y, n-1-z), n))
        result.append(coordinates_to_number((x, n-1-y, z), n))
        result.append(coordinates_to_number((x, n-1-y, n-1-z), n))
        result.append(coordinates_to_number((n-1-x, y, z), n))
        result.append(coordinates_to_number((n-1-x, y, n-1-z), n))
        result.append(coordinates_to_number((n-1-x, n-1-y, z), n))
        result.append(coordinates_to_number((n-1-x, n-1-y, n-1-z), n))
    
    return result

def coordinates_to_number(a, n):
    result = a[0] * n**2 + a[1] * n + a[2]
    return result