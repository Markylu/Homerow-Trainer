from config import config
import sqlalchemy as db
from sqlalchemy.orm import declarative_base, sessionmaker

params = config()
engine = db.create_engine(f"postgresql://{params['user']}:{params['password']}@{params['host']}:5432/{params['database']}")
Base = declarative_base()
Session = sessionmaker(bind=engine)

class Input(Base):
    __tablename__ = 'inputs'
    id = db.Column(db.Integer, primary_key=True)
    keystroke = db.Column(db.String(20))
    timestamp = db.Column(db.String(100))
 

    def __repr__(self):
        return f'{self.keystroke} ({self.timestamp})'
    
    def to_tuple(self):
        return (self.id, self.keystroke, self.timestamp)

def get_inputs():
    with Session() as session:
        result = session.query(Input).all()
        return [input.to_tuple() for input in result]
    
def add_input(keystroke, timestamp):
    with Session() as session:
        new_input = Input(keystroke=keystroke, timestamp=timestamp)
        session.add(new_input)
        session.commit()
        print(f"Added input: {keystroke} ({timestamp})")