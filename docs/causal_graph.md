# Causal Graph Appendix

```mermaid
graph LR
    Education[Education / Education Proxy] --> Allocation[Task Allocation]
    Education --> Outcome[Task Success / Outcome]
    WorkerContext[Experience / Geography / Route Context] --> Allocation
    WorkerContext --> Outcome
    Platform[Platform Rules / Incentives] --> Allocation
    Platform --> Outcome
    Allocation --> Outcome
```