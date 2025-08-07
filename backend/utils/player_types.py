OUTFIELD_POSITIONS = ['OF']
INFIELD_POSITIONS = ['1B', '2B', '3B', 'SS']
PITCHER_POSITIONS = ['LHP', 'RHP']

class PlayerType:
    """ 
    Base class for player types in baseball recruitment.
    This class serves as a foundation for defining specific player types.

    This will be the parent class to all player types, such as Outfielder, Infielder, etc.
    Mandatory attributes include:
    - height: int -> Height of the player in inches
    - weight: int -> Weight of the player in pounds
    - primary_position: str -> Primary position of the player (e.g., 'OF', '1B', etc.)
    - region (str): Region of the player (e.g., 'West', 'South', 'Northeast', 'Midwest)

    These attributes are used for every player type, so it is defined here.
    """
    def __init__(self, height: int, weight: int, primary_position: str, region: str):
        self.height = height
        self.weight = weight
        self.region = region
        self.primary_position = primary_position

    def get_player_type(self):
        """
        Returns the type of player based on primary position.
        """
        return self.__class__.__name__

class PlayerCatcher(PlayerType):
    def __init__(
        self,
        height: int,
        weight: int,
        primary_position: str,
        hitting_handedness: str,
        throwing_hand: str,
        region: str,
        exit_velo_max: float,
        c_velo: float,
        pop_time: float
    ):
        """
        Initialize a PlayerCatcher object.

        Parameters:
        height (int): Height of the player in inches
        weight (int): Weight of the player in pounds
        primary_position (str): Primary position of the player (e.g., 'C')
        hitting_handedness (str): Hitting handedness of the player (e.g., 'R', 'L', 'S')
        throwing_hand (str): Throwing hand of the player (e.g., 'L', 'R')
        region (str): Region of the player (e.g., 'West', 'South', 'Northeast', 'Midwest)
        c_velo (float): Catcher velocity (mph)
        pop_time (float): Pop time of the player (seconds)
        exit_velo_max (float): Maximum exit velocity (mph)
        """
        super().__init__(height, weight, region, primary_position)
        self.inf_velo = c_velo
        self.pop_time = pop_time
        self.exit_velo_max = exit_velo_max
        self.hitting_handedness = hitting_handedness
        self.throwing_hand = throwing_hand
    
    def get_player_type(self) -> str:
        return "Catcher"


class PlayerInfielder(PlayerType):
    def __init__(
        self,
        height: int,
        weight: int,
        primary_position: str,
        hitting_handedness: str,
        throwing_hand: str,
        region: str,
        exit_velo_max: float,
        inf_velo: float
    ):
        """
        Initialize a PlayerInfielder object.

        Parameters:
        height (int): Height of the player in inches
        weight (int): Weight of the player in pounds
        primary_position (str): Primary position of the player (e.g., '1B', '2B', '3B', 'SS')
        hitting_handedness (str): Hitting handedness of the player (e.g., 'R', 'L', 'S')
        throwing_hand (str): Throwing hand of the player (e.g., 'L', 'R')
        region (str): Region of the player (e.g., 'West', 'South', 'Northeast', 'Midwest)
        inf_velo (float): Infield velocity (mph)
        exit_velo_max (float): Maximum exit velocity (mph)
        """
        super().__init__(height, weight, region, primary_position)
        self.inf_velo = inf_velo
        self.exit_velo_max = exit_velo_max
        self.hitting_handedness = hitting_handedness
        self.throwing_hand = throwing_hand

    def get_player_type(self) -> str:
        return "Infielder"


class PlayerOutfielder(PlayerType):
    def __init__(
        self,
        height: int,
        weight: int,
        primary_position: str,
        hitting_handedness: str,
        throwing_hand: str,
        region: str,
        exit_velo_max: float,
        of_velo: float,
    ):
        """
        Initialize a PlayerOutfielder object.

        Parameters:
        height (int): Height of the player in inches
        weight (int): Weight of the player in pounds
        primary_position (str): Primary position of the player (e.g., 'OF')
        hitting_handedness (str): Hitting handedness of the player (e.g., 'R', 'L', 'S')
        throwing_hand (str): Throwing hand of the player (e.g., 'L', 'R')
        region (str): Region of the player (e.g., 'West', 'South', 'Northeast', 'Midwest)
        of_velo (float): Outfield velocity (mph)
        exit_velo_max (float): Maximum exit velocity (mph)
        """
        super().__init__(height, weight, region, primary_position)
        self.of_velo = of_velo
        self.exit_velo_max = exit_velo_max
        self.hitting_handedness = hitting_handedness
        self.throwing_hand = throwing_hand

    def get_player_type(self) -> str:
        return "Outfielder"


class PlayerPitcher(PlayerType):
    # TODO: go through and add base attributes for pitchers
    def __init__(
            self,
            height: int,
            weight: int,
            primary_position: str,
            throwing_hand: str,
            region: str
        ):
        super().__init__(height, weight, region, primary_position)
        self.throwing_hand = throwing_hand

    def get_player_type(self) -> str:
        return "Pitcher"


def create_player(height: int, weight: int, primary_position: str, hitting_handedness: str, throwing_hand: str) -> PlayerType:
    """
    Factory method to create the appropriate PlayerType subclass based on the player's position.
    """
    if primary_position == 'C':
        return PlayerCatcher(height, weight, primary_position, hitting_handedness, throwing_hand)
    elif primary_position in INFIELD_POSITIONS:
        return PlayerInfielder(height, weight, primary_position, hitting_handedness, throwing_hand)
    elif primary_position in OUTFIELD_POSITIONS:
        return PlayerOutfielder(height, weight, primary_position, hitting_handedness, throwing_hand)
    elif primary_position in PITCHER_POSITIONS:
        return PlayerPitcher(height, weight, primary_position, hitting_handedness, throwing_hand)
    else:
        raise ValueError(f"Invalid primary position: {primary_position}")