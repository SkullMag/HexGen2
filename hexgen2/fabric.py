"""Inspect or explicitly provision the two-A30 pilot; never delete/extend slices."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import ipaddress
import json
from pathlib import Path
import sys


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def available(host, key):
    capacity, allocated = host[f"{key}_capacity"], host[f"{key}_allocated"]
    return min(capacity - allocated, host.get(f"{key}_available", capacity - allocated))


def inspect_plan(fab, plan):
    resources = fab.get_resources(update=True, force_refresh=True)
    snapshots = []
    if len(plan["nodes"]) != 2 or len({n["host"] for n in plan["nodes"]}) != 2:
        raise ValueError("Two distinct physical hosts required")
    if len({n["site"] for n in plan["nodes"]}) != 1:
        raise ValueError("This L2Bridge pilot requires one site")
    for spec in plan["nodes"]:
        if [spec[k] for k in ("cores", "ram", "disk")] != [8, 64, 100]:
            raise ValueError("This pilot's explicit flavor is 8 cores / 64 GiB / 100 GB")
        site = resources.get_site(spec["site"])
        host = resources.get_hosts_by_site(spec["site"])[spec["host"]]
        gpu = host["components"]["GPU-A30"]
        fits = (site["state"] in {"Active", "PartMaint"} and host["state"] == "Active"
                and min(gpu["capacity"] - gpu["allocated"], gpu.get("available", gpu["capacity"])) >= 1
                and all(available(host, key) >= spec[key] for key in ("cores", "ram", "disk")))
        snapshots.append({"requested": spec, "advertised": host, "fits": fits})
    return {"checked_utc": datetime.now(timezone.utc).isoformat(), "reserved": False,
            "fits": all(h["fits"] for h in snapshots), "hosts": snapshots}


def capture(sl, plan):
    sl.update()
    nodes = []
    for node in sl.get_nodes():
        spec = next(n for n in plan["nodes"] if n["name"] == node.get_name())
        interface = node.get_component("data").get_interfaces()[0]
        fields = {k: str(node.get_fim().get_property(k)) for k in
                  ("capacities", "capacity_allocations", "capacity_hints", "labels", "label_allocations")}
        nodes.append({"name": node.get_name(), "role": spec["role"], "rank": spec["rank"],
                      "site": node.get_site(), "physical_host": node.get_host(), "image": node.get_image(),
                      "cores": node.get_cores(), "ram_gib": node.get_ram(), "disk_gb": node.get_disk(),
                      "reservation_state": str(node.get_reservation_state()), "fim": fields,
                      "components": [{"name": c.get_name(), "model": c.get_model(),
                                      "labels": str(c.get_fim().get_property("labels")),
                                      "label_allocations": str(c.get_fim().get_property("label_allocations"))}
                                     for c in node.get_components()],
                      "interface": interface.get_device_name(), "actual_addresses": interface.get_ip_addr(),
                      "configured_data_address": spec["address"], "tp": 1, "pp": 1})
    return {"captured_utc": datetime.now(timezone.utc).isoformat(), "kind": "actual-fabric-allocation",
            "slice_id": sl.get_slice_id(), "slice_name": sl.get_name(), "state": sl.get_state(),
            "lease_end": str(sl.get_lease_end()), "nodes": nodes, "network": plan["network"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "create", "status"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fabric-rc", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--name", default="hexgen2-pilot-20261019")
    parser.add_argument("--lease-hours", type=int, default=4)
    args = parser.parse_args()
    from fabrictestbed_extensions.fablib.fablib import FablibManager
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(sys.stderr):
        fab = FablibManager(fabric_rc=str(args.fabric_rc), auto_token_refresh=False, log_level="ERROR",
                            data_dir=str(args.state.parent / "fablib-cache"))
    plan = json.loads(args.plan.read_text())
    if args.action == "plan":
        print(json.dumps(inspect_plan(fab, plan), indent=2, default=str))
        return
    if args.action == "status":
        state = json.loads(args.state.read_text())
        result = capture(fab.get_slice(slice_id=state["slice_id"]), plan)
        save(args.state.parent / "allocation.json", result)
        print(json.dumps(result, indent=2, default=str))
        return
    if args.state.exists() or any(s.get_name() == args.name for s in fab.get_slices()):
        raise FileExistsError("Slice/state already exists; inspect with status")
    if not 1 <= args.lease_hours <= 12:
        raise ValueError("Pilot lease must be between 1 and 12 hours")
    availability = inspect_plan(fab, plan)
    save(args.state.parent / "availability-before.json", availability)
    if not availability["fits"]:
        raise ValueError("Requested hosts no longer fit; inspect a new plan")
    sl = fab.new_slice(name=args.name)
    ports = []
    for spec in plan["nodes"]:
        node = sl.add_node(name=spec["name"], site=spec["site"], host=spec["host"],
                           instance_type="fabric.c8.m64.d100", image="docker_ubuntu_22")
        node.add_component(model="GPU_A30", name="gpu")
        ports.extend(node.add_component(model="NIC_Basic", name="data").get_interfaces())
    sl.add_l2network(name="hexgen2-data", type="L2Bridge", interfaces=ports)
    state = {"status": "submitting", "plan": plan, "lease_hours": args.lease_hours}
    save(args.state, state)
    try:
        returned = sl.submit(wait=False, lease_in_hours=args.lease_hours)
        state.update(slice_id=sl.get_slice_id() or returned, status="submitted")
        save(args.state, state)
        sl.wait(timeout=900, progress=False)
        sl.update()
        if sl.get_state() != "StableOK":
            raise RuntimeError("Slice is not StableOK")
        sl.wait_ssh(timeout=900, progress=False)
        sl.post_boot_config()
        for spec in plan["nodes"]:
            interface = sl.get_node(spec["name"]).get_component("data").get_interfaces()[0]
            interface.ip_link_up()
            interface.ip_addr_add(addr=ipaddress.ip_address(spec["address"]),
                                  subnet=ipaddress.ip_network(plan["network"]["subnet"]))
        state["status"] = "ready"
    except BaseException as error:
        state.update(status="incomplete", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(args.state, state)
    result = capture(sl, plan)
    save(args.state.parent / "allocation.json", result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
