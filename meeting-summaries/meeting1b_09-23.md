# Meeting on 23rd

#### Location: DSI
#### People: Cynthia, Josh, Srinidhi, Zach, 

## Points discussed- Necessary Features

### Mouse Features: Species, Parent, Generation, Genome, Size

### Features from Video:

#### Basic- Sequence Data:

1. Where is the mouse? - Position with Time
2. Velocity/Acceleration with Time

#### Derived- Aggregation and Snippets:

1. Time: %T_Inside, %T_Outside, %T_Quadrant, %T_safety, %T_peeking, %T(Active)
2. Position: Avg(Position in each quadrant), Avg(Manhattan Distance) from safe spot
3. Velocity: In each region, Region -> Region
4. Preference of Left turns vs Right turns
5. Region-> Region frequency
6. Feature changes with time

### Identified Bottlenecks

1. Understanding the Data Structure/Schema of data (Inconsistent data format inside folders ->  How to go forward?)
2. Google drive mounted on Colab -> Timed out (Should try removing videos)
3. Area of mouse at the ends: shadow based or movement based?
4. Should ask for data in a tabular format (Time Series of Position, Velocity & Acceleration)
