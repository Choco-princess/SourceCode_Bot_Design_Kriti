#include <Servo.h>

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

// Camera Servo
#define SERVO_PIN A4 
Servo camServo;

// Timers
unsigned long last_ping_time = 0;
unsigned long last_photo_time = 0; // Prevents taking 100 photos of the same card

// ---------------- PID & MOTOR VARIABLES ----------------
float Kp = 50.0;   // Start between 40 - 80
float Kd = 500.0;  // Start between 300 - 800

int base_speed = 255; 
int turn_speed = 170; 
int max_speed = 255;

#define turn_delay 10  
#define stop_timer 30  

float error = 0, previous_error = 0;
float center_point = 2.5; 
int turn_value = 0;

// ---------------- STRATEGY VARIABLES ----------------
bool obstacle_hit = false;       // True when it bounces off the obstacle
bool returned_to_start = false;  // True when it successfully returns to base
bool isRunning = false;          // Controlled by Pi via START_RUN / STOP

#define PI_DONE_TIMEOUT 5000     // ms to wait for Pi's DONE before giving up

void setup() {
  Serial.begin(115200);

  // 1. Initialize Camera Servo FIRST
  Serial.println("Initializing Camera Servo...");
  camServo.attach(SERVO_PIN);
  camServo.write(90); // Snap exactly to center
  delay(500);         // Let the physical servo settle

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

  // Wait for Pi to send START_RUN
  Serial.println("Waiting for START_RUN from Pi...");
  while (!isRunning) {
    if (Serial.available() > 0) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();
      if (cmd == "START_RUN") {
        isRunning = true;
      }
    }
  }

  // Drive off the start box
  Serial.println("START_RUN received. Leaving start box...");
  while (digitalRead(IR1) == 1 || digitalRead(IR4) == 1) {
    driveMotors(base_speed, base_speed);
  }
}

// ==========================================
// WAIT FOR Pi DONE SIGNAL
// ==========================================
// Called after servo has aimed camera at the card.
// Sends IMAGE_READY, then blocks until Pi replies DONE (or times out).
void waitForPiDone() {
  Serial.println("IMAGE_READY");
  unsigned long startTime = millis();

  while (millis() - startTime < PI_DONE_TIMEOUT) {
    if (Serial.available() > 0) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();
      if (cmd == "DONE") {
        return;                 // Pi finished — resume driving
      } else if (cmd == "STOP") {
        isRunning = false;      // Emergency stop from Pi
        driveMotors(0, 0);
        return;
      }
    }
  }

  // Pi didn't respond in time
  Serial.println("PI_TIMEOUT");
}

void loop() {
  // Check for Serial Commands from Pi
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "START_RUN") {
      isRunning = true;
    } else if (cmd == "STOP") {
      isRunning = false;
      driveMotors(0, 0);
    }
  }

  if (!isRunning) {
    driveMotors(0, 0);
    return;
  }

  // 1. NON-BLOCKING SENSOR SWEEP (Once every 50ms)
  if (millis() - last_ping_time > 50) {
    last_ping_time = millis(); 

    // ---------------------------------------------------------
    // PHASE 1: OBSTACLE HUNTING (Before we hit it)
    // ---------------------------------------------------------
    if (!obstacle_hit) {
      int dF = getDistance(TRIG_F, ECHO_F);
      
      if (dF > 0 && dF <= 20) {
        Serial.println("Obstacle Detected FRONT! Executing LEFT U-Turn...");
        executeObstacleUTurn(1); 
        obstacle_hit = true; 
      }
      else {
        delay(5); // Acoustic breathing room
        int dL = getDistance(TRIG_L, ECHO_L);
        
        if (dL > 0 && dL <= 20) {
          Serial.println("Obstacle Detected LEFT! Executing RIGHT U-Turn...");
          executeObstacleUTurn(2); 
          obstacle_hit = true; 
        }
        else {
          delay(5); 
          int dR = getDistance(TRIG_R, ECHO_R);
          
          if (dR > 0 && dR <= 20) {
            Serial.println("Obstacle Detected RIGHT! Executing LEFT U-Turn...");
            executeObstacleUTurn(1); 
            obstacle_hit = true; 
          }
        }
      }
    }
    
    // ---------------------------------------------------------
    // PHASE 2: IMAGE CARD HUNTING (After returning to start)
    // ---------------------------------------------------------
    // Wait at least 2000ms after the LAST photo to scan again, preventing stutter loops
    else if (returned_to_start && (millis() - last_photo_time > 2000)) {
      int dL = getDistance(TRIG_L, ECHO_L);
      delay(5); // Acoustic breathing room
      int dR = getDistance(TRIG_R, ECHO_R);
      
      // Check Left Side (10cm to 30cm)
      if (dL >= 10 && dL <= 30) {
        Serial.println("Card Detected LEFT! Stopping to scan...");
        driveMotors(0, 0);   // BRAKE
        delay(300);          // Let chassis settle to prevent blurry photos
        camServo.write(180); // Snap camera left
        delay(500);          // Let servo fully rotate and settle
        waitForPiDone();     // Send IMAGE_READY, wait for Pi's DONE
        camServo.write(90);  // Snap back to forward
        delay(300);          // Let servo settle before driving
        last_photo_time = millis(); // Trigger cooldown timer so we can drive past this card
        
        // Wipe PID memory to prevent jerking when motors resume
        error = 0; previous_error = 0; turn_value = 0; 
      }
      // Check Right Side (10cm to 30cm)
      else if (dR >= 10 && dR <= 30) {
        Serial.println("Card Detected RIGHT! Stopping to scan...");
        driveMotors(0, 0);   // BRAKE
        delay(300);          
        camServo.write(0);   // Snap camera right
        delay(500);          // Let servo fully rotate and settle
        waitForPiDone();     // Send IMAGE_READY, wait for Pi's DONE
        camServo.write(90);  // Snap back to forward
        delay(300);          // Let servo settle before driving
        last_photo_time = millis(); 
        
        error = 0; previous_error = 0; turn_value = 0;
      }
    }
  }

  // 2. RUN HIGH-SPEED PID (Continuous)
  Line_Follow();
}

// ==========================================
// DIRECTIONAL OBSTACLE ESCAPE MANEUVER
// ==========================================
void executeObstacleUTurn(int direction) {
  driveMotors(0, 0);
  delay(300);

  if (direction == 1) {
    driveMotors(-turn_speed, turn_speed); // Spin Left
  } else {
    driveMotors(turn_speed, -turn_speed); // Spin Right
  }
  
  delay(600); 

  while (digitalRead(IR2) == 0 && digitalRead(IR3) == 0) {
    if (direction == 1) {
      driveMotors(-turn_speed, turn_speed);
    } else {
      driveMotors(turn_speed, -turn_speed);
    }
  }

  driveMotors(0, 0);
  delay(200);

  error = 0; previous_error = 0; turn_value = 0;
}

// ==========================================
// 180-DEGREE START-POINT RESTART MANEUVER
// ==========================================
void executeStartUTurn() {
  driveMotors(0, 0);
  delay(300);

  driveMotors(-turn_speed, turn_speed);
  
  delay(600); // FIXED to 600ms so it doesn't over-spin!

  while (digitalRead(IR2) == 0 && digitalRead(IR3) == 0) {
    driveMotors(-turn_speed, turn_speed);
  }

  driveMotors(0, 0);
  delay(200);

  error = 0; previous_error = 0; turn_value = 0;
}

// ==========================================
// PID LINE FOLLOWING & END DETECTION
// ==========================================
void Line_Follow() {
  int s1 = digitalRead(IR1);
  int s2 = digitalRead(IR2);
  int s3 = digitalRead(IR3);
  int s4 = digitalRead(IR4);

  int sensor_sum = s1 + s2 + s3 + s4;
  float calculated_pos = center_point; 

  if (sensor_sum > 0) {
    int line_position = (s1 * 1) + (s2 * 2) + (s3 * 3) + (s4 * 4);
    calculated_pos = (float)line_position / (float)sensor_sum;
  }

  error = center_point - calculated_pos;
  int PID = (error * Kp) + (Kd * (error - previous_error));
  previous_error = error;

  int left_motor = base_speed - PID;  
  int right_motor = base_speed + PID; 

  driveMotors(left_motor, right_motor);

  if (s1 == 1 && s4 == 0) turn_value = 1;
  if (s4 == 1 && s1 == 0) turn_value = 2;

  if (sensor_sum == 0) {
    if (turn_value == 1) { 
      delay(turn_delay);
      driveMotors(-turn_speed, turn_speed);
      while (digitalRead(IR2) == 0 && digitalRead(IR3) == 0) {}
      turn_value = 0;
    }
    else if (turn_value == 2) {
      delay(turn_delay);
      driveMotors(turn_speed, -turn_speed);
      while (digitalRead(IR2) == 0 && digitalRead(IR3) == 0) {}
      turn_value = 0;
    }
  }
  
  // --- START BOX / FINISH LINE DETECTION ---
  else if (sensor_sum == 4) {
    delay(stop_timer); 
    
    if (digitalRead(IR1) == 1 && digitalRead(IR2) == 1 && digitalRead(IR3) == 1 && digitalRead(IR4) == 1) {
      
      // Scenario A: We hit the obstacle and returned to the Start Box
      if (obstacle_hit == true && returned_to_start == false) {
        Serial.println("Back at Start Point! Executing U-Turn...");
        executeStartUTurn();
        returned_to_start = true; 
      } 
      // Catch-all Termination: We finished the run OR had a perfectly clean run with no obstacles
      else {
        driveMotors(0, 0); 
        Serial.println("RUN_COMPLETE");
        isRunning = false;
        while (true) {} 
      }
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
  
  long duration = pulseIn(echoPin, HIGH, 3000); 
  if (duration == 0) return 999; 
  return duration * 0.034 / 2;
}
