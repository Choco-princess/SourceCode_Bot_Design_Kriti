// ---------------- PIN DEFINITIONS ----------------
#define IR1 2   // Far Left
#define IR2 3   // Mid Left
#define IR3 4   // Mid Right
#define IR4 5   // Far Right

#define ENA 6     // PWM Left
#define IN1 7
#define IN2 8
#define IN3 9
#define IN4 10
#define ENB 11    // PWM Right

// Ultrasonic Pins
#define TRIG_F 12
#define ECHO_F 13
#define TRIG_L A0
#define ECHO_L A1
#define TRIG_R A2
#define ECHO_R A3

// ---------------- PID & MOTOR VARIABLES ----------------
float Kp = 50.0;   
float Kd = 500.0;  

int base_speed = 180; 
int turn_speed = 170; 
int max_speed = 255;

// Tuning Delays
#define turn_delay 10  
#define stop_timer 30  

// Obstacle Avoidance Tuning
int turn_90_delay = 700;     // Time needed to turn exactly 90 degrees
int intersection_ignore = 400; // ms to ignore intersection after turning (avoids re-detecting the original +)

float error = 0, previous_error = 0;
float center_point = 2.5; 
int turn_value = 0;
bool isRunning = false;
unsigned long lastCaptureTime = 0;

void setup() {
  Serial.begin(115200);

  // Configure IR Pins
  pinMode(IR1, INPUT); pinMode(IR2, INPUT);
  pinMode(IR3, INPUT); pinMode(IR4, INPUT);

  // Configure Motor Pins
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT); pinMode(ENB, OUTPUT);

  // Configure Ultrasonic Pins
  pinMode(TRIG_F, OUTPUT); pinMode(ECHO_F, INPUT);
  pinMode(TRIG_L, OUTPUT); pinMode(ECHO_L, INPUT);
  pinMode(TRIG_R, OUTPUT); pinMode(ECHO_R, INPUT);
}

void loop() {
  // Check for Serial Commands from Pi
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "START_RUN") {
      isRunning = true;
      lastCaptureTime = millis();
    } else if (cmd == "STOP") {
      isRunning = false;
      driveMotors(0, 0);
    }
  }

  if (!isRunning) {
    driveMotors(0, 0);
    return;
  }

  // Auto-capture every 10 seconds while running
  if (millis() - lastCaptureTime >= 10000) {
    Serial.println("IMAGE_READY");
    lastCaptureTime = millis();
  }

  // 1. Check Front Ultrasonic First
  int distFront = getDistance(TRIG_F, ECHO_F);

  // 2. Obstacle Trigger (Detects the box at <= 25cm away)
  if (distFront > 0 && distFront <= 25) {
    Serial.println("OBSTACLE_DETECTED");
    Serial.println("Obstacle Detected! Executing Active Bypass...");
    dodgeObstacle();
  }
  // 3. Early braking zone (25-40 cm): slow down so we can stop in time
  else if (distFront > 25 && distFront <= 40) {
    int slow_speed = base_speed / 2;
    driveMotors(slow_speed, slow_speed);
  }
  else {
    // 4. Normal Operation
    Line_Follow();
  }
}

// ==========================================
// + INTERSECTION OBSTACLE AVOIDANCE
// ==========================================
// The track provides a + (plus) intersection before each obstacle.
// The perpendicular line of the + connects to the adjacent parallel
// track (500-600mm away). Instead of wall-hugging, the robot:
//   1. Turns onto the perpendicular connector line
//   2. Follows it using the same PID as normal line-following
//   3. Turns onto the next parallel track at the T-intersection
// This is more reliable because it uses proven PID line-following
// rather than blind driving or ultrasonic wall-tracking.

// Follow the perpendicular connector line using PID until the robot
// hits the next parallel track (detected as intersection: 3+ sensors on black).
void followLineUntilIntersection() {
  float line_err = 0, line_prev_err = 0;
  unsigned long startTime = millis();

  Serial.println("  PID following connector line...");

  while (true) {
    int s1 = digitalRead(IR1);
    int s2 = digitalRead(IR2);
    int s3 = digitalRead(IR3);
    int s4 = digitalRead(IR4);
    int sensor_sum = s1 + s2 + s3 + s4;

    // Wait at least intersection_ignore ms before checking for the next
    // intersection, so we don't re-detect the original + we just turned from.
    bool pastOriginal = (millis() - startTime > (unsigned long)intersection_ignore);

    // Intersection detected: 3+ sensors on black means we've hit the next track
    if (pastOriginal && sensor_sum >= 3) {
      Serial.println("  Next track intersection detected!");
      driveMotors(0, 0);
      break;
    }

    // PID line following on the perpendicular connector line
    float calculated_pos = center_point;
    if (sensor_sum > 0) {
      int line_position = (s1 * 1) + (s2 * 2) + (s3 * 3) + (s4 * 4);
      calculated_pos = (float)line_position / (float)sensor_sum;
    }

    line_err = center_point - calculated_pos;
    int PID = (line_err * Kp) + (Kd * (line_err - line_prev_err));
    line_prev_err = line_err;

    int leftSpeed  = base_speed - PID;
    int rightSpeed = base_speed + PID;
    driveMotors(leftSpeed, rightSpeed);

    // Safety timeout
    if (millis() - startTime > 5000) {
      Serial.println("  Connector line follow timeout!");
      driveMotors(0, 0);
      break;
    }
  }
  delay(200);
}

void dodgeObstacle() {
  // 1. HARD STOP
  driveMotors(0, 0);
  delay(300);

  // 2. DECIDE DIRECTION — check which side has more room
  int distL = getDistance(TRIG_L, ECHO_L);
  delay(30);
  int distR = getDistance(TRIG_R, ECHO_R);

  Serial.print("Left dist: "); Serial.print(distL);
  Serial.print("  Right dist: "); Serial.println(distR);

  bool goLeft = (distL >= distR);

  if (goLeft) {
    Serial.println(">>> Turning LEFT onto + connector");
  } else {
    Serial.println(">>> Turning RIGHT onto + connector");
  }

  // 3. FIRST 90° TURN — turn onto the perpendicular connector line of the +
  if (goLeft) {
    driveMotors(-turn_speed, turn_speed);   // pivot left
  } else {
    driveMotors(turn_speed, -turn_speed);   // pivot right
  }
  delay(turn_90_delay);
  driveMotors(0, 0);
  delay(200);

  // 4. PID-FOLLOW THE CONNECTOR LINE to the next parallel track
  Serial.println("Following connector line to next track...");
  followLineUntilIntersection();

  // 5. SECOND 90° TURN — same direction as first turn
  //    On a serpentine track, adjacent lines run opposite directions.
  //    Turning the same way twice (e.g. left-left or right-right)
  //    puts the robot heading the correct way on the next track.
  Serial.println("Turning onto next track...");
  if (goLeft) {
    driveMotors(-turn_speed, turn_speed);   // pivot left again
  } else {
    driveMotors(turn_speed, -turn_speed);   // pivot right again
  }
  delay(turn_90_delay);
  driveMotors(0, 0);
  delay(200);

  // 6. Reset PID memory to prevent a violent jerk
  error = 0;
  previous_error = 0;
  turn_value = 0;

  Serial.println("Obstacle bypassed via + intersection. Resuming line follow.");
}

// ==========================================
// PID LINE FOLLOWING
// ==========================================
void Line_Follow() {
  // 1. READ SENSORS
  int s1 = digitalRead(IR1);
  int s2 = digitalRead(IR2);
  int s3 = digitalRead(IR3);
  int s4 = digitalRead(IR4);

  int sensor_sum = s1 + s2 + s3 + s4;
  float calculated_pos = center_point; 

  // 2. WEIGHTED AVERAGE MATH
  if (sensor_sum > 0) {
    int line_position = (s1 * 1) + (s2 * 2) + (s3 * 3) + (s4 * 4);
    calculated_pos = (float)line_position / (float)sensor_sum;
  }

  // 3. PID CALCULATION
  error = center_point - calculated_pos;
  int PID = (error * Kp) + (Kd * (error - previous_error));
  previous_error = error;

  // 4. APPLY TO MOTORS
  int left_motor = base_speed - PID;  
  int right_motor = base_speed + PID; 

  driveMotors(left_motor, right_motor);

  // 5. TURN MEMORY DETECTION
  if (s1 == 1 && s4 == 0) turn_value = 1;
  if (s4 == 1 && s1 == 0) turn_value = 2;

  // 6. TURN EXECUTION (When all sensors are on white)
  if (sensor_sum == 0) {
    // Left Turn Execution
    if (turn_value == 1) { 
      delay(turn_delay);
      driveMotors(-turn_speed, turn_speed);
      while (digitalRead(IR2) == 0 && digitalRead(IR3) == 0) {}
      turn_value = 0;
    }
    // Right Turn Execution
    else if (turn_value == 2) {
      delay(turn_delay);
      driveMotors(turn_speed, -turn_speed);
      while (digitalRead(IR2) == 0 && digitalRead(IR3) == 0) {}
      turn_value = 0;
    }
  }

  // 7. STOP DETECTION (When all sensors are on black)
  else if (sensor_sum == 4) {
    delay(stop_timer); 
    if (digitalRead(IR1) == 1 && digitalRead(IR2) == 1 && digitalRead(IR3) == 1 && digitalRead(IR4) == 1) {
      driveMotors(0, 0); // STOP
      Serial.println("DONE");
      while (true) {} // Lock in an infinite loop
    }
  }
}

// ==========================================
// MOTOR CONTROL
// ==========================================
void driveMotors(int leftSpeed, int rightSpeed) {
  if (leftSpeed >= 0) {
    digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  } else {
    digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
    leftSpeed = -leftSpeed; 
  }

  if (rightSpeed >= 0) {
    digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  } else {
    digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
    rightSpeed = -rightSpeed; 
  }

  leftSpeed = constrain(leftSpeed, 0, max_speed);
  rightSpeed = constrain(rightSpeed, 0, max_speed);

  analogWrite(ENA, leftSpeed);
  analogWrite(ENB, rightSpeed);
}

// ==========================================
// ULTRASONIC SENSOR LOGIC
// ==========================================
int getDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW); 
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH); 
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // 6000us timeout (approx 100cm) so the PID loop stays lightning fast
  long duration = pulseIn(echoPin, HIGH, 6000); 
  
  if (duration == 0) return 999; 
  return duration * 0.034 / 2;
}