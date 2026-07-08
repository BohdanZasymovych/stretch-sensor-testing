#define READ_PIN_INITIAL A1
#define READ_PIN_DISTORTED A5

void setup() {
  Serial.begin(2000000);
  pinMode(READ_PIN_INITIAL, INPUT);
  pinMode(READ_PIN_DISTORTED, INPUT);
}

void loop() {
  uint16_t raw_data_initial = analogRead(READ_PIN_INITIAL);
  uint16_t raw_data_distorted = analogRead(READ_PIN_DISTORTED);

  Serial.print(micros());
  Serial.print(",");
  Serial.print(raw_data_initial);
  Serial.print(",");
  Serial.print(raw_data_distorted);
  Serial.println();
}