
int odorA=3, odorB=4, odorC=5, odorD=6;
char byteread;

void setup() {
  // put your setup code here, to run once:
  pinMode(odorA, OUTPUT);
  pinMode(odorB, OUTPUT);
  pinMode(odorC, OUTPUT);
  pinMode(odorD, OUTPUT);
  Serial.begin(9600);
  digitalWrite(odorA,LOW);
  digitalWrite(odorB,LOW);
  digitalWrite(odorC,LOW);
  digitalWrite(odorD,LOW);

}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()){
    byteread=(char)Serial.read();
    switch (byteread){
      case ('A'):
        digitalWrite(odorA, HIGH);
        Serial.println ("odor A is on");
        break;
     case ('a'):
        digitalWrite(odorA, LOW);
        Serial.println ("odor A is off");
        break;
        case ('B'):
        digitalWrite(odorB, HIGH);
        Serial.println ("odor B is on");
        break;
     case ('b'):
        digitalWrite(odorB, LOW);
        Serial.println ("odor B is off");
        break;
        case ('C'):
        digitalWrite(odorC, HIGH);
        Serial.println ("odor C is on");
        break;
     case ('c'):
        digitalWrite(odorC, LOW);
        Serial.println ("odor C is off");
        break;
        case ('D'):
        digitalWrite(odorD, HIGH);
        Serial.println ("odor D is on");
        break;
     case ('d'):
        digitalWrite(odorD, LOW);
        Serial.println ("odor D is off");
        break;
    }
  }

}
