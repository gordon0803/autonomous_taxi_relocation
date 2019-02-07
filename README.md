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

5. Update waiting time, determine if existing passenger will leave, and generate new passengers

6. Serve passengers as FIFO system

7. Summarize the status of the system (states, rewards)


Done. 

# Todo list (2019/2/5)
## 1. Change the code to incorporate the training of multiple stations ............ DONE
## 2. Do we need to see the entire system states? Do we need to relocate to nearby stations only?
## 3. Mechanism to compare not relocation vs relocation
## 4. Develop a small testcase with 10 taxi stands, this would require a much smaller convolution network structure

# Todo List (2019/2/6)
## Change experience replay and each station has a distinct experience buffer. .........DONE

