#define BASE_VOLTAGE_V 5.0
#define RESISTANCE_OM 1000.0
#define MEASUREMENT_PIN A0

void setup() {
  Serial.begin(9600);
}

void loop() {
  int rawVal = analogRead(MEASUREMENT_PIN);

  double voltageDrop = (rawVal * BASE_VOLTAGE_V) / 1024.0;
  
  double currentAmps = voltageDrop / RESISTANCE_OM;

  double currentMA = currentAmps * 1000.0;

  Serial.print("Current (mA): ");
  Serial.println(currentMA, 2);

  delay(200); 
}