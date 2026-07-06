void setup() {
  Serial.begin(9600);
}

void loop() {
  for (size_t i=0; i<100; ++i) {
    Serial.println((float)i/100);
    delay(10);
  }
  for (size_t i=0; i<100; ++i) {
    Serial.println((float)(100-i)/100);
    delay(10);
  }
}
