# Hyperledger Fabric Test Network

This directory contains the Hyperledger Fabric blockchain network infrastructure that provides the distributed ledger foundation for the BPFL system, including smart contract deployment, identity management, and network orchestration.

## Directory Structure

```
test-network/
├── network.sh                    # Main network management script
├── start.sh                     # Quick start script for BPFL
├── req.sh                       # Wallet initialization script
├── setOrgEnv.sh                 # Organization environment setup
├── monitordocker.sh             # Docker container monitoring
├── network.config               # Network configuration file
├── README.md                    # Hyperledger Fabric documentation
├── CHAINCODE_AS_A_SERVICE_TUTORIAL.md  # Chaincode deployment guide
│
├── configtx/                    # Channel configuration
│   └── configtx.yaml           # Channel and genesis block configuration
│
├── organizations/               # Certificate Authority and crypto materials
│   ├── ccp-generate.sh         # Connection profile generator
│   ├── ccp-template.json       # JSON connection profile template
│   ├── ccp-template.yaml       # YAML connection profile template
│   ├── cryptogen/              # Cryptographic material generation
│   ├── cfssl/                  # Alternative CA setup
│   └── fabric-ca/              # Certificate Authority configuration
│
├── compose/                     # Docker Compose configurations
│   ├── compose-test-net.yaml   # Main network containers
│   ├── compose-ca.yaml         # Certificate Authority containers
│   ├── compose-couch.yaml      # CouchDB state database
│   ├── compose-bft-test-net.yaml # Byzantine Fault Tolerance setup
│   ├── docker/                 # Docker-specific configurations
│   └── podman/                 # Podman container alternatives
│
├── scripts/                     # Automation and utility scripts
│   ├── createChannel.sh        # Channel creation automation
│   ├── deployCC.sh             # Chaincode deployment
│   ├── deployCCAAS.sh          # Chaincode-as-a-Service deployment
│   ├── ccutils.sh              # Chaincode utility functions
│   ├── configUpdate.sh         # Configuration update utilities
│   ├── envVar.sh               # Environment variable setup
│   ├── orderer.sh              # Orderer node management
│   ├── packageCC.sh            # Chaincode packaging
│   ├── pkgcc.sh                # Alternative packaging script
│   ├── setAnchorPeer.sh        # Anchor peer configuration
│   ├── utils.sh                # General utility functions
│   └── org3-scripts/           # Third organization scripts
│
├── addOrg3/                     # Dynamic organization addition
│   ├── addOrg3.sh              # Organization addition script
│   ├── configtx.yaml           # Extended configuration
│   ├── org3-crypto.yaml        # Org3 cryptographic configuration
│   ├── ccp-generate.sh         # Org3 connection profiles
│   ├── compose/                # Org3 Docker configurations
│   └── fabric-ca/              # Org3 Certificate Authority
│
├── bft-config/                  # Byzantine Fault Tolerance
│   └── configtx.yaml          # BFT-specific configuration
│
└── prometheus-grafana/          # Monitoring and metrics
    ├── docker-compose.yaml     # Monitoring stack
    ├── prometheus/             # Prometheus configuration
    ├── grafana/                # Grafana dashboards
    └── grafana_db/             # Grafana database
```

## Network Architecture

### 🏗️ **Blockchain Infrastructure**

The BPFL system operates on a **3-organization Hyperledger Fabric network**:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Organization 1 │    │   Organization 2 │    │   Orderer Org   │
│                  │    │                  │    │                  │
│ • peer0.org1     │    │ • peer0.org2     │    │ • orderer.example│
│ • CouchDB        │    │ • CouchDB        │    │ • Consensus      │
│ • Certificate CA │    │ • Certificate CA │    │ • Block Creation │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 🔗 **Network Components**

#### **Peer Nodes:**
- **peer0.org1.example.com**: Primary endorsing peer for Org1
- **peer0.org2.example.com**: Primary endorsing peer for Org2
- **Endorsement Policy**: Requires signatures from both organizations

#### **Orderer Service:**
- **orderer.example.com**: Raft-based ordering service
- **Consensus**: Crash Fault Tolerant (CFT) by default
- **BFT Option**: Byzantine Fault Tolerant consensus available

#### **Certificate Authorities:**
- **ca.org1.example.com**: Issues certificates for Org1 members
- **ca.org2.example.com**: Issues certificates for Org2 members
- **TLS CA**: Separate CA for TLS certificates

#### **State Database:**
- **CouchDB**: JSON document database for rich queries
- **LevelDB**: Alternative key-value state database
- **Data Persistence**: Volume-mounted for container restarts

### 📊 **Channel Configuration**

#### **Main Channel**: `main`
```yaml
Channel: main
Organizations:
  - Org1MSP (peer0.org1.example.com)
  - Org2MSP (peer0.org2.example.com)
Orderers:
  - orderer.example.com
Chaincode:
  - tokenCC (Token management)
  - modelCC (Model submissions)
```

#### **Endorsement Policies:**
```yaml
# Both organizations must endorse transactions
Endorsement: "AND('Org1MSP.peer', 'Org2MSP.peer')"

# Majority endorsement for high availability
# Endorsement: "OutOf(2, 'Org1MSP.peer', 'Org2MSP.peer')"
```

## Quick Start for BPFL

### 🚀 **Automated Setup** (`start.sh`)

```bash
#!/bin/bash
# Optimized network startup for BPFL system

# 1. Bring up the network with CA and CouchDB
./network.sh up createChannel -ca -s couchdb

# 2. Deploy token management chaincode
./network.sh deployCC -ccn tokenCC -ccp ../token-transfer/token-transfer-chaincode/ -ccl javascript

# 3. Deploy model management chaincode  
./network.sh deployCC -ccn modelCC -ccp ../model-propose/model-propose-chaincode/ -ccl javascript

echo "BPFL network ready for federated learning!"
```

### 🎯 **Initialize BPFL System** (`req.sh`)

```bash
#!/bin/bash
# Initialize BPFL-specific blockchain state

# Set environment for Org1
source ./setOrgEnv.sh 1

echo "Initializing BPFL token system..."
peer chaincode invoke -o orderer.example.com:7050 --tls \
  --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
  -C main -n tokenCC \
  -c '{"function":"InitWallets","Args":["4","50","10","300"]}'

echo "Initializing model management..."
peer chaincode invoke -o orderer.example.com:7050 --tls \
  --cafile ${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
  -C main -n modelCC \
  -c '{"function":"InitRoundInfo","Args":["4"]}'

echo "BPFL blockchain initialization complete!"
```

## Running the Standard Test Network

You can use the `./network.sh` script to stand up a simple Fabric test network. The test network has two peer organizations with one peer each and a single node raft ordering service. You can also use the `./network.sh` script to create channels and deploy chaincode. For more information, see [Using the Fabric test network](https://hyperledger-fabric.readthedocs.io/en/latest/test_network.html). The test network is being introduced in Fabric v2.0 as the long term replacement for the `first-network` sample.

If you are planning to run the test network with consensus type BFT then please pass `-bft` flag as input to the `network.sh` script when creating the channel. Note that currently this sample does not yet support the use of consensus type BFT and CA together.
That is to create a network use:
```bash
./network.sh up -bft
```

To create a channel use:

```bash
./network.sh createChannel -bft
```

To restart a running network use:

```bash
./network.sh restart -bft
```

Note that running the createChannel command will start the network, if it is not already running.

Before you can deploy the test network, you need to follow the instructions to [Install the Samples, Binaries and Docker Images](https://hyperledger-fabric.readthedocs.io/en/latest/install.html) in the Hyperledger Fabric documentation.

## Using the Peer commands

The `setOrgEnv.sh` script can be used to set up the environment variables for the organizations, this will help to be able to use the `peer` commands directly.

First, ensure that the peer binaries are on your path, and the Fabric Config path is set assuming that you're in the `test-network` directory.

```bash
 export PATH=$PATH:$(realpath ../bin)
 export FABRIC_CFG_PATH=$(realpath ../config)
```

You can then set up the environment variables for each organization. The `./setOrgEnv.sh` command is designed to be run as follows.

```bash
export $(./setOrgEnv.sh Org2 | xargs)
```

(Note bash v4 is required for the scripts.)

You will now be able to run the `peer` commands in the context of Org2. If a different command prompt, you can run the same command with Org1 instead.
The `setOrgEnv` script outputs a series of `<name>=<value>` strings. These can then be fed into the export command for your current shell.

## Chaincode-as-a-service

To learn more about how to use the improvements to the Chaincode-as-a-service please see this [tutorial](./test-network/../CHAINCODE_AS_A_SERVICE_TUTORIAL.md). It is expected that this will move to augment the tutorial in the [Hyperledger Fabric ReadTheDocs](https://hyperledger-fabric.readthedocs.io/en/release-2.4/cc_service.html)


## Podman

*Note - podman support should be considered experimental but the following has been reported to work with podman 4.1.1 on Mac. If you wish to use podman a LinuxVM is recommended.*

Fabric's `install-fabric.sh` script has been enhanced to support using `podman` to pull down images and tag them rather than docker. The images are the same, just pulled differently. Simply specify the 'podman' argument when running the `install-fabric.sh` script. 

Similarly, the `network.sh` script has been enhanced so that it can use `podman` and `podman-compose` instead of docker. Just set the environment variable `CONTAINER_CLI` to `podman` before running the `network.sh` script:

```bash
CONTAINER_CLI=podman ./network.sh up
````

As there is no Docker-Daemon when using podman, only the `./network.sh deployCCAAS` command will work. Following the Chaincode-as-a-service Tutorial above should work. 


