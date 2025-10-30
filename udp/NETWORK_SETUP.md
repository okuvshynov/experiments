# Network Setup for Link Aggregation Testing

This guide explains how to set up two Cat6A ethernet connections between two macOS machines for UDP latency benchmarking with link aggregation.

## Network Topology

```
Machine A (Client)                    Machine B (Server)
┌─────────────────┐                  ┌─────────────────┐
│                 │                  │                 │
│  en0: 10.0.0.1  ├──────────────────┤  en0: 10.0.0.2  │
│                 │   Cat6A Cable 1  │                 │
│                 │                  │                 │
│  en1: 10.0.1.1  ├──────────────────┤  en1: 10.0.1.2  │
│                 │   Cat6A Cable 2  │                 │
└─────────────────┘                  └─────────────────┘
```

## Prerequisites

- Two macOS machines with at least 2 ethernet ports each
- Two Cat6A ethernet cables
- Direct connection (no switch/router between machines)
- Administrative privileges (sudo access)

## Step 1: Identify Network Interfaces

On both machines, list all network interfaces:

```bash
ifconfig -a | grep "^en"
```

You should see interfaces like `en0`, `en1`, `en2`, etc. In this guide, we assume:
- **Link 1** uses `en0`
- **Link 2** uses `en1`

## Step 2: Configure Machine A (Client)

On the client machine, configure both interfaces with static IPs:

```bash
# Configure first link (en0)
sudo ifconfig en0 10.0.0.1 netmask 255.255.255.0 up

# Configure second link (en1)
sudo ifconfig en1 10.0.1.1 netmask 255.255.255.0 up
```

Verify configuration:

```bash
ifconfig en0 | grep "inet "
ifconfig en1 | grep "inet "
```

Expected output:
```
inet 10.0.0.1 netmask 0xffffff00 broadcast 10.0.0.255
inet 10.0.1.1 netmask 0xffffff00 broadcast 10.0.1.255
```

## Step 3: Configure Machine B (Server)

On the server machine, configure both interfaces:

```bash
# Configure first link (en0)
sudo ifconfig en0 10.0.0.2 netmask 255.255.255.0 up

# Configure second link (en1)
sudo ifconfig en1 10.0.1.2 netmask 255.255.255.0 up
```

Verify configuration:

```bash
ifconfig en0 | grep "inet "
ifconfig en1 | grep "inet "
```

Expected output:
```
inet 10.0.0.2 netmask 0xffffff00 broadcast 10.0.0.255
inet 10.0.1.2 netmask 0xffffff00 broadcast 10.0.1.255
```

## Step 4: Test Connectivity

From Machine A (client), test both links:

```bash
# Test link 1
ping -c 4 10.0.0.2

# Test link 2
ping -c 4 10.0.1.2
```

Both pings should succeed with low latency (typically < 1ms for direct connection).

## Step 5: Optional - Set Interface Speed/Duplex

For best performance, you can force 10 Gigabit mode (if supported):

```bash
# On both machines, for each interface
sudo ifconfig en0 media 10GbaseT mediaopt full-duplex
sudo ifconfig en1 media 10GbaseT mediaopt full-duplex
```

Or for 1 Gigabit:

```bash
sudo ifconfig en0 media 1000baseT mediaopt full-duplex
sudo ifconfig en1 media 1000baseT mediaopt full-duplex
```

Check current media settings:

```bash
ifconfig en0 | grep media
ifconfig en1 | grep media
```

## Step 6: Run the Benchmarks

### On Machine B (Server)

Start the server with sudo (required for high priority):

```bash
cd /Users/oleksandr/projects/experiments/udp
sudo ./build/udpserver_aggr
```

Expected output:
```
[+] Process priority set to highest
[+] Link 0 listening on port 4321
[+] Link 1 listening on port 4322

[*] UDP server with 2 links ready
[*] Press Ctrl+C to stop
```

### On Machine A (Client)

Start the client, providing both server IP addresses:

```bash
cd /Users/oleksandr/projects/experiments/udp

# Using default 20KB packet size (5120 floats)
sudo ./build/udpclient_aggr 10.0.0.2 10.0.1.2

# Or specify custom packet size (e.g., 8192 bytes)
sudo ./build/udpclient_aggr 10.0.0.2 10.0.1.2 8192
```

Expected output:
```
[+] Process priority set to highest
[+] Link 0 initialized: 10.0.0.2:4321 -> local port 65400
[+] Link 1 initialized: 10.0.1.2:4322 -> local port 65401

[*] Link aggregation active with 2 links
[*] Packet size: 20480 bytes (20.00 KB)
[*] BUSY POLLING MODE - will use 100% of one CPU core
[*] Press Ctrl+C to stop

Overall: pps= 5234 avg= 191.32us min= 145.67us max= 312.45us stdev= 23.12us
Link 0 : pps= 2617 avg= 190.88us min= 145.67us max= 298.34us stdev= 22.87us
Link 1 : pps= 2617 avg= 191.76us min= 148.23us max= 312.45us stdev= 23.39us
Bandwidth: 856.32 Mbps (107.04 MB/s)
```

## Troubleshooting

### No connectivity on one or both links

1. Check physical cable connections
2. Verify interfaces are up:
   ```bash
   ifconfig en0 | grep "status"
   ifconfig en1 | grep "status"
   ```
   Should show `status: active`

3. Check if interfaces have correct IPs:
   ```bash
   ifconfig en0
   ifconfig en1
   ```

### "Address already in use" error

The port is already in use. Either:
- Kill the existing process: `sudo killall udpserver_aggr` or `sudo killall udpclient_aggr`
- Wait a few seconds for OS to release the port

### Poor performance or high latency

1. Disable WiFi/Bluetooth to reduce interference
2. Check for CPU throttling or power saving modes
3. Verify you're using Cat6A or better cables
4. Make sure cables are directly connected (no switch/hub)
5. Run with sudo for high priority scheduling

### One link has much higher latency

1. Check cable quality - try swapping cables
2. Verify both interfaces have same speed/duplex settings
3. Check for ethernet adapter issues with:
   ```bash
   netstat -i
   ```
   Look for errors/collisions on the interfaces

## Resetting Configuration

To remove the static IP configuration and return to DHCP:

```bash
# On both machines
sudo ifconfig en0 down
sudo ifconfig en1 down

# Let the system reconfigure (or restart network service)
sudo ifconfig en0 up
sudo ifconfig en1 up
```

Or simply disconnect the cables and the interfaces will go back to their previous state after a network restart.

## Making Configuration Persistent (Optional)

The `ifconfig` settings are temporary and will be lost on reboot. To make them persistent:

1. Create a launch daemon plist file (advanced)
2. Use a network management tool
3. Or simply run the ifconfig commands again after each reboot

For testing purposes, temporary configuration is usually sufficient.
