# 2024 F1 Driver-Constructor Mapping
# Maps driver names to IDs and embeds with constructors

DRIVERS_2024 = {
    # Existing IDs (in CSV historical data) - Updated for 2026 season
    1: {"name": "Lewis Hamilton", "constructor_id": 6},  # Ferrari
    4: {"name": "Fernando Alonso", "constructor_id": 117},  # Aston Martin
    807: {"name": "Nico Hülkenberg", "constructor_id": 215},  # Audi
    815: {"name": "Sergio Pérez", "constructor_id": 216},  # Cadillac
    822: {"name": "Valtteri Bottas", "constructor_id": 216},  # Cadillac
    830: {"name": "Max Verstappen", "constructor_id": 9},  # Red Bull
    832: {"name": "Carlos Sainz", "constructor_id": 3},  # Williams
    839: {"name": "Esteban Ocon", "constructor_id": 210},  # Haas
    840: {"name": "Lance Stroll", "constructor_id": 117},  # Aston Martin
    842: {"name": "Pierre Gasly", "constructor_id": 214},  # Alpine
    844: {"name": "Charles Leclerc", "constructor_id": 6},  # Ferrari
    846: {"name": "Lando Norris", "constructor_id": 1},  # McLaren
    847: {"name": "George Russell", "constructor_id": 131},  # Mercedes
    848: {"name": "Alexander Albon", "constructor_id": 3},  # Williams
    
    # New 2026 drivers (not in historical CSV, use synthetic IDs 900+)
    900: {"name": "Franco Colapinto", "constructor_id": 214},  # Alpine
    901: {"name": "Gabriel Bortoleto", "constructor_id": 215},  # Audi
    902: {"name": "Oliver Bearman", "constructor_id": 210},  # Haas
    903: {"name": "Oscar Piastri", "constructor_id": 1},  # McLaren
    904: {"name": "Andrea Kimi Antonelli", "constructor_id": 131},  # Mercedes
    905: {"name": "Liam Lawson", "constructor_id": 217},  # Racing Bulls
    906: {"name": "Arvid Lindblad", "constructor_id": 217},  # Racing Bulls
    907: {"name": "Isack Hadjar", "constructor_id": 9},  # Red Bull
}

CONSTRUCTORS_2024 = {
    1: "McLaren",
    3: "Williams",
    6: "Ferrari",
    9: "Red Bull",
    117: "Aston Martin",
    131: "Mercedes",
    210: "Haas",
    214: "Alpine",
    # N/A constructors for 2024
    215: "Audi",
    216: "Cadillac",
    217: "Racing Bulls",
}

CIRCUITS_2024 = {
    1: {"name": "Albert Park", "location": "Melbourne, Australia"},
    3: {"name": "Bahrain", "location": "Sakhir, Bahrain"},
    4: {"name": "Barcelona", "location": "Montmeló, Spain"},
    6: {"name": "Monaco", "location": "Monte-Carlo, Monaco"},
    7: {"name": "Montreal", "location": "Montreal, Canada"},
    9: {"name": "Silverstone", "location": "Silverstone, UK"},
    11: {"name": "Hungaroring", "location": "Budapest, Hungary"},
    13: {"name": "Spa", "location": "Spa, Belgium"},
    14: {"name": "Monza", "location": "Monza, Italy"},
    15: {"name": "Marina Bay", "location": "Singapore"},
    17: {"name": "Shanghai", "location": "Shanghai, China"},
    18: {"name": "Interlagos", "location": "São Paulo, Brazil"},
    21: {"name": "Imola", "location": "Imola, Italy"},
    22: {"name": "Suzuka", "location": "Suzuka, Japan"},
    24: {"name": "Abu Dhabi", "location": "Abu Dhabi, UAE"},
    32: {"name": "Mexico City", "location": "Mexico City, Mexico"},
    39: {"name": "Zandvoort", "location": "Zandvoort, Netherlands"},
    69: {"name": "Austin", "location": "Austin, USA"},
    70: {"name": "Spielberg", "location": "Spielberg, Austria"},
    73: {"name": "Baku", "location": "Baku, Azerbaijan"},
    77: {"name": "Jeddah", "location": "Jeddah, Saudi Arabia"},
    78: {"name": "Lusail", "location": "Lusail, Qatar"},
    79: {"name": "Miami", "location": "Miami, USA"},
}

# Helper function to get sorted 2024 grid
def get_2024_grid():
    """Returns list of (driver_id, driver_name, constructor_name)"""
    grid = []
    for driver_id in sorted(DRIVERS_2024.keys()):
        driver_info = DRIVERS_2024[driver_id]
        driver_name = driver_info["name"]
        constructor_id = driver_info["constructor_id"]
        constructor_name = CONSTRUCTORS_2024.get(constructor_id, "Unknown")
        grid.append({
            "driver_id": driver_id,
            "driver_name": driver_name,
            "constructor_id": constructor_id,
            "constructor_name": constructor_name
        })
    return grid
