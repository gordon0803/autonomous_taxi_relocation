# autonomous_taxi_relocation

This is the repo for the autonomous taxi relocation project.

We will follow the following steps in each time step of taxi simulation:

1. Execute taxi relocation strategy, send taxis to relocation queue

2. Taxi in travel, time+=-1

3. Check the charging status of taxis at each taxi station.
If full charged, send back to queue
Else, send back to the charging queue

4. Check the status of taxi arrival
If time_to_destination<=0, check the battery of this taxi.
  - If the battery > 20% capacity, send to service queue
  - Else, send to charging queue
  Mark the taxi as arrived

Else send the taxi back to travel queue

5. Update waiting time, and generate passengers

6. Serve passengers

7. Summarize the status of the system (states, rewards)


Done. 
