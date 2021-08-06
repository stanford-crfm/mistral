Training A Model With GCP + Kubernetes
======================================

This tutorial will walk through training a model on Google Cloud with `Kubernetes <https://kubernetes.io/>`_.

Preliminaries
-------------

We will assume you have a `Google Cloud <https://cloud.google.com/>`_ account set up already.

For this tutorial you will need to install the `gcloud <https://cloud.google.com/sdk/docs/downloads-interactive>`_ and `kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/>`_ command line utilities.

Creating A Kubernetes Cluster
-----------------------------

We will now create a basic Kubernetes cluster with

* 2 main machines for managing the overall cluster
* A node pool that will create GPU machines when jobs are submitted
* A 1 TB persistent volume the machines can use for data storage

This tutorial describes the Kubernetes set up we used when training models, but of course you can customize this set up as you see fit for your situation.

On the Google Cloud Console, go to the "Kubernetes Engine (Clusters)" page. Click on "CREATE".

Choose the "GKE Standard" option.

On the "Cluster basics" page, set the name of your cluster and choose the zone you want for your cluster.
You will want to choose a zone with A100 machines such as ``us-central-1a``. In our working example, we
are calling the cluster "tutorial-cluster".

In the "NODE POOLS" section, change the name of the default pool to "main". Click on "Nodes" and change
the machine type to "e2-standard".

When finished, click "CREATE" at the bottom and the Kubernetes cluster will be created.

Adding A Node Pool To Your Cluster
----------------------------------

When the cluster has finished, you can click on its name and see cluster info. Click on "NODES". You
will be brought to a page that shows the node pools for the cluster and the nodes.

At the top of the screen click on "ADD NODE POOL". Set the name of the node pool to "node-1". Set the number of nodes
to 0 and check "Enable autoscaling". With autoscaling, Kubernetes will launch nodes when you submit jobs. When there
are no active jobs, you will have no active machines running. When a job is submitted, the node pool will scale up to
meet the needs of the job. Set the minimum number of nodes to 0 and set the maximum to the maximum number of GPU machines
you want running at any given time.

Click on "Nodes" on the left sidebar, and customize the types of machines the node pool will use. This tutorial will
assume you are running on NVIDIA Tesla A100's with 8 GPUs, and the default machine configuration.

When finished, click "CREATE". You should see "pool-1" show up in your list of node pools.

Creating The Persistent Volume
------------------------------

The next step is to create the persistent volume. We will create a 1 TB volume, though you may want more space.

You will need to have installed ``gcloud`` and ``kubectl``. Instructions for installing them can be found in
the "Preliminaries" section above.

First create the disk: ::

    gcloud compute disks create --size=1000GB --zone=us-central1-a --type pd-ssd pd-tutorial

Then set up the nfs server (from the ``gcp`` directory in the mistral repo): ::

    kubectl apply -f nfs/nfs_server.yaml
    kubectl apply -f nfs/nfs_service.yaml
    kubectl get services

You should see output like this: ::

    NAME         TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)                      AGE
    kubernetes   ClusterIP   10.48.0.1      <none>        443/TCP                      135m
    nfs-server   ClusterIP   10.48.14.252   <none>        2049/TCP,20048/TCP,111/TCP   11s

Extract the IP address for the nfs-server (10.48.14.252 in the example output), and update the ``nfs/nfs_pv.yaml``
file. Then run: ::

    kubectl apply -f nfs_pv.yaml

You should see output like: ::

    NAME                          READY   STATUS    RESTARTS   AGE
    nfs-server-697fbd7f8d-pvsdb   1/1     Running   0          14m

The persistent volume should now be ready for usage.

Installing Drivers
------------------

Run this command to set up the GPU drivers. If you do not run this command, nodes will be unable to use GPUs. ::

    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
